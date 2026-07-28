import './globals.css';
import { AuthProvider } from '@/lib/auth-context';

export const metadata = {
  // Lets pages declare relative Open Graph URLs; without it Next.js warns and
  // social/link previews fall back to a bare URL.
  metadataBase: new URL('https://web-ahmadiss12s-projects.vercel.app'),
  title: 'CoachVision Dashboard',
  description: 'Trainer and admin dashboard for CoachVision',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
