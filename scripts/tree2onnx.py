"""Convert a fitted sklearn GradientBoostingClassifier into an ONNX graph that
uses only standard tensor ops (Gather, LessOrEqual, MatMul, Equal, ...).

onnxruntime-web ships without the ai.onnx.ml domain (TreeEnsembleClassifier),
so we compile the trees to the "GEMM" form used by Microsoft's Hummingbird:
for each tree, a leaf is reached iff the number of satisfied left-turn
conditions on its path equals the number of left turns on that path.
Output: 'label' (int64) and 'probabilities' (float32), like skl2onnx.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def compile_gbdt(model, n_features: int, path: str):
    est = model.estimators_  # [n_stages, K] of DecisionTreeRegressor
    n_stages, K = est.shape
    trees = [est[s, k].tree_ for s in range(n_stages) for k in range(K)]
    tree_class = np.array([k for s in range(n_stages) for k in range(K)])
    T = len(trees)
    I = max(int((t.children_left != -1).sum()) for t in trees) or 1
    L = max(int((t.children_left == -1).sum()) for t in trees)

    feat = np.zeros((T, I), np.int64)
    thr = np.full((T, I), np.float32(np.inf), np.float32)
    C = np.zeros((T, I, L), np.float32)
    D = np.full((T, L), -1.0, np.float32)
    V = np.zeros((T, L), np.float32)
    for ti, t in enumerate(trees):
        internal = [n for n in range(t.node_count) if t.children_left[n] != -1]
        leaves = [n for n in range(t.node_count) if t.children_left[n] == -1]
        ii = {n: i for i, n in enumerate(internal)}
        li = {n: i for i, n in enumerate(leaves)}
        for n in internal:
            feat[ti, ii[n]] = t.feature[n]
            # sklearn goes left when x <= threshold (thresholds are float64; X is float32)
            thr[ti, ii[n]] = np.float32(t.threshold[n])
        # walk paths
        stack = [(0, [])]
        while stack:
            n, trail = stack.pop()
            if t.children_left[n] == -1:
                l = li[n]
                lefts = 0
                for (a, went_left) in trail:
                    C[ti, ii[a], l] = 1.0 if went_left else -1.0
                    lefts += went_left
                D[ti, l] = lefts
                V[ti, l] = t.value[n].ravel()[0]
            else:
                stack.append((t.children_left[n], trail + [(n, 1)]))
                stack.append((t.children_right[n], trail + [(n, 0)]))

    lr = model.learning_rate
    W = np.zeros((T, K), np.float32)
    W[np.arange(T), tree_class] = lr
    # initial raw prediction (log-odds / log-priors) from the init estimator
    init = model._raw_predict_init(np.zeros((1, n_features), np.float32)).astype(np.float32).ravel()

    inits = [
        numpy_helper.from_array(feat.reshape(-1), 'feat'),
        numpy_helper.from_array(thr.reshape(1, -1), 'thr'),
        numpy_helper.from_array(C, 'C'),
        numpy_helper.from_array(D.reshape(T, 1, L), 'D'),
        numpy_helper.from_array(V.reshape(T, 1, L), 'V'),
        numpy_helper.from_array(W, 'W'),
        numpy_helper.from_array(init, 'init'),
        numpy_helper.from_array(np.array([0, T, I], np.int64), 'shp3'),
        numpy_helper.from_array(np.array([2], np.int64), 'ax2'),
    ]
    nodes = [
        helper.make_node('Gather', ['X', 'feat'], ['xg'], axis=1),                 # [N, T*I]
        helper.make_node('LessOrEqual', ['xg', 'thr'], ['le']),                      # go left?
        helper.make_node('Cast', ['le'], ['lef'], to=TensorProto.FLOAT),
        helper.make_node('Reshape', ['lef', 'shp3'], ['le3']),                       # [N, T, I]
        helper.make_node('Transpose', ['le3'], ['leT'], perm=[1, 0, 2]),             # [T, N, I]
        helper.make_node('MatMul', ['leT', 'C'], ['score']),                         # [T, N, L]
        helper.make_node('Equal', ['score', 'D'], ['hit']),
        helper.make_node('Cast', ['hit'], ['hitf'], to=TensorProto.FLOAT),
        helper.make_node('Mul', ['hitf', 'V'], ['val']),
        helper.make_node('ReduceSum', ['val', 'ax2'], ['tv'], keepdims=0),           # [T, N]
        helper.make_node('Transpose', ['tv'], ['tvT'], perm=[1, 0]),                 # [N, T]
        helper.make_node('MatMul', ['tvT', 'W'], ['raw0']),                          # [N, K]
        helper.make_node('Add', ['raw0', 'init'], ['raw']),
    ]
    if K == 1:  # binary: sigmoid -> [1-p, p]
        nodes += [
            helper.make_node('Sigmoid', ['raw'], ['p1']),
            helper.make_node('Neg', ['p1'], ['np1']),
            helper.make_node('Add', ['np1', 'one'], ['p0']),
            helper.make_node('Concat', ['p0', 'p1'], ['probabilities'], axis=1),
        ]
        inits.append(numpy_helper.from_array(np.array([1.0], np.float32), 'one'))
    else:
        nodes += [helper.make_node('Softmax', ['raw'], ['probabilities'], axis=1)]
    nodes += [helper.make_node('ArgMax', ['probabilities'], ['label'], axis=1, keepdims=0)]

    graph = helper.make_graph(
        nodes, 'gbdt_gemm',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, n_features])],
        [helper.make_tensor_value_info('label', TensorProto.INT64, [None]),
         helper.make_tensor_value_info('probabilities', TensorProto.FLOAT, [None, max(K, 2)])],
        inits,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], producer_name='formflow-tree2onnx')
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    return T, I, L
