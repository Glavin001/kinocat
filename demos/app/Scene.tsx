'use client';

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { createHelloCube, KINOCAT_THREE_VERSION } from '@kinocat/three';

export default function Scene() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.1,
      100,
    );
    camera.position.set(2, 2, 3);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Imported across the monorepo workspace boundary from @kinocat/three.
    const cube = createHelloCube();
    scene.add(cube);

    let frameId = 0;
    const animate = () => {
      cube.rotation.x += 0.01;
      cube.rotation.y += 0.013;
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(frameId);
      renderer.dispose();
      cube.geometry.dispose();
      (cube.material as THREE.Material).dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <main style={{ position: 'relative', width: '100vw', height: '100vh' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      <div
        style={{
          position: 'absolute',
          top: 16,
          left: 16,
          color: '#e6e6f0',
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
          fontSize: 13,
          lineHeight: 1.5,
          pointerEvents: 'none',
        }}
      >
        <div>kinocat monorepo foundation</div>
        <div style={{ opacity: 0.6 }}>
          @kinocat/three v{KINOCAT_THREE_VERSION}
        </div>
      </div>
    </main>
  );
}
