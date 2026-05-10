import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'PharmaGuard | Shelf Monitoring',
  description: 'Precision Zero-Shot Drug Placement Detection',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen bg-slate-50">
        <nav className="bg-white border-b border-slate-200 px-8 py-4 flex items-center justify-between sticky top-0 z-50">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-[#002147] rounded-md flex items-center justify-center font-bold text-white text-xs">
              PG
            </div>
            <span className="text-lg font-bold tracking-tight text-[#002147]">PharmaGuard</span>
          </div>
          <div className="flex gap-8 text-[13px] font-semibold text-slate-500 uppercase tracking-wider">
            <a href="/" className="hover:text-[#002147] transition-colors">Dashboard</a>
          </div>
        </nav>
        <main className="max-w-[1400px] mx-auto p-8">
          {children}
        </main>
      </body>
    </html>
  );
}
