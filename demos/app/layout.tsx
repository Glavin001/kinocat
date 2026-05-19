import type { ReactNode } from 'react';

export const metadata = {
  title: 'kinocat demos',
  description: 'In-browser IGHA* time-extended kinodynamic planning',
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover' as const,
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          background: '#0b0b0f',
          overflowX: 'hidden',
          WebkitTextSizeAdjust: '100%',
        }}
      >
        {children}
      </body>
    </html>
  );
}
