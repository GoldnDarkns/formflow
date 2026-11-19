// The living background: soft aurora glows in the tray colours, and over them
// a flow field, thin streams of light that drift across the screen like pages
// moving through a sorter. The streams speed up while a document is being
// read and bend gently around the pointer. Drawn on one canvas; paused when
// the tab is hidden and replaced by a still picture for reduced motion.
import { useEffect, useRef } from 'react';

const COLORS = ['--t-passport', '--t-emirates_id', '--t-driving_licence', '--t-visa', '--t-id_card', '--marker'];

interface P { x: number; y: number; life: number; c: number; w: number }

export function Backdrop({ busy }: { busy: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const busyRef = useRef(busy);
  busyRef.current = busy;

  useEffect(() => {
    const cv = ref.current!;
    const ctx = cv.getContext('2d')!;
    const reduce = matchMedia('(prefers-reduced-motion: reduce)');
    const dark = matchMedia('(prefers-color-scheme: dark)');
    let W = 0, H = 0, dpr = 1, raf = 0, t = 0, speed = 1, last = performance.now();
    let colors: string[] = [];
    let parts: P[] = [];
    const mouse = { x: -9999, y: -9999 };

    const readColors = () => {
      const cs = getComputedStyle(document.documentElement);
      colors = COLORS.map((v) => cs.getPropertyValue(v).trim() || '#8493ff');
    };
    const spawn = (p: P) => {
      p.x = Math.random() * W; p.y = Math.random() * H;
      p.life = 80 + Math.random() * 220;
      p.c = Math.floor(Math.random() * colors.length);
      p.w = 0.6 + Math.random() * 1.3;
    };
    const resize = () => {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      W = window.innerWidth; H = window.innerHeight;
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // fewer streams on small screens, so phones stay cool
      const n = Math.round(Math.min(340, Math.max(70, (W * H) / 5200)));
      parts = Array.from({ length: n }, () => { const p = { x: 0, y: 0, life: 0, c: 0, w: 1 }; spawn(p); return p; });
    };
    // the field: a slow, swirling direction for every point on the screen
    const angle = (x: number, y: number) => {
      const k = 0.0021;
      return (Math.sin(x * k + t * 0.21) + Math.cos(y * k * 1.3 - t * 0.17) + Math.sin((x + y) * k * 0.6 + t * 0.11)) * 1.25 + 0.35;
    };
    const step = (dt: number) => {
      // ease towards the target speed: brisk while reading, calm otherwise
      speed += ((busyRef.current ? 2.6 : 1) - speed) * 0.04;
      t += dt * 0.001 * speed;
      // fade what is already drawn, towards transparent, so the glows underneath stay visible
      ctx.globalCompositeOperation = 'destination-out';
      ctx.fillStyle = `rgba(0,0,0,${dark.matches ? 0.032 : 0.06})`;
      ctx.fillRect(0, 0, W, H);
      ctx.globalCompositeOperation = dark.matches ? 'lighter' : 'source-over';
      ctx.lineCap = 'round';
      const v = 0.055 * dt * speed;
      for (const p of parts) {
        const a = angle(p.x, p.y);
        let dx = Math.cos(a) * v, dy = Math.sin(a) * v;
        // streams part around the pointer
        const mx = p.x - mouse.x, my = p.y - mouse.y, d2 = mx * mx + my * my;
        if (d2 < 26000) { const f = (1 - d2 / 26000) * 1.6; const d = Math.sqrt(d2) || 1; dx += (mx / d) * f; dy += (my / d) * f; }
        const nx = p.x + dx, ny = p.y + dy;
        ctx.globalAlpha = Math.min(1, p.life / 60) * (dark.matches ? 0.62 : 0.4) * (busyRef.current ? 1.25 : 1);
        ctx.strokeStyle = colors[p.c];
        ctx.lineWidth = p.w;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(nx, ny); ctx.stroke();
        p.x = nx; p.y = ny; p.life -= speed;
        if (p.life <= 0 || nx < -20 || ny < -20 || nx > W + 20 || ny > H + 20) spawn(p);
      }
      ctx.globalAlpha = 1;
    };
    const frame = (now: number) => {
      const dt = Math.min(48, now - last);
      last = now;
      step(dt);
      raf = requestAnimationFrame(frame);
    };
    const start = () => {
      cancelAnimationFrame(raf);
      ctx.clearRect(0, 0, W, H);
      if (reduce.matches) { for (let i = 0; i < 160; i++) step(16); return; } // a still picture
      last = performance.now();
      raf = requestAnimationFrame(frame);
    };
    const onVis = () => (document.hidden ? cancelAnimationFrame(raf) : start());
    const onMove = (e: PointerEvent) => { mouse.x = e.clientX; mouse.y = e.clientY; };
    const onLeave = () => { mouse.x = mouse.y = -9999; };
    const onScheme = () => { readColors(); start(); };

    readColors(); resize(); start();
    const onResize = () => { resize(); start(); };
    window.addEventListener('resize', onResize);
    window.addEventListener('pointermove', onMove, { passive: true });
    document.addEventListener('pointerleave', onLeave);
    document.addEventListener('visibilitychange', onVis);
    dark.addEventListener('change', onScheme);
    reduce.addEventListener('change', start);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      window.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerleave', onLeave);
      document.removeEventListener('visibilitychange', onVis);
      dark.removeEventListener('change', onScheme);
      reduce.removeEventListener('change', start);
    };
  }, []);

  return (
    <div className={'backdrop' + (busy ? ' busy' : '')} aria-hidden>
      <i className="glow g1" /><i className="glow g2" /><i className="glow g3" /><i className="glow g4" />
      <canvas ref={ref} />
      <i className="grain" />
    </div>
  );
}
