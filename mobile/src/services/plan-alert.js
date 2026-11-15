import { Alert } from 'react-native';
import { isPlanRequiredError } from './plan-access';

/**
 * If `error` is the server refusing a request because of the account's plan,
 * show its message with a way to the plans screen and return true.
 */
export function alertIfPlanRequired(error, router) {
  if (!isPlanRequiredError(error)) {
    return false;
  }
  Alert.alert('Not included in your plan', error.message, [
    { text: 'Not now', style: 'cancel' },
    { text: 'See plans', onPress: () => router.push('/(app)/plans') },
  ]);
  return true;
}
