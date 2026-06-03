import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import { getDailyReport } from '../api/reports';
import { buildDailyReportHtml } from './daily-report-html';

export async function exportDailyReportPdf({ date } = {}) {
  const report = await getDailyReport({ date });
  const html = buildDailyReportHtml(report);
  const result = await Print.printToFileAsync({ html });

  const canShare = await Sharing.isAvailableAsync();
  if (canShare) {
    await Sharing.shareAsync(result.uri, {
      mimeType: 'application/pdf',
      dialogTitle: 'Share CoachVision daily report',
      UTI: 'com.adobe.pdf',
    });
  }

  return {
    report,
    uri: result.uri,
    shared: canShare,
  };
}
