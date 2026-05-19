import type { ReactNode } from 'react';

export const metadata = {
  title: 'kinocat demos',
  description: 'In-browser IGHA* time-extended kinodynamic planning',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, background: '#0b0b0f' }}>{children}</body>
    </html>
  );
}
