import './globals.css';
import { AuthProvider } from '@/lib/auth-context';

export const metadata = {
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
