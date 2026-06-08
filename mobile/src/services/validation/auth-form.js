const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function validateEmailInput(email) {
  const value = String(email ?? '').trim();
  if (!value) {
    return 'Email is required.';
  }
  if (!EMAIL_PATTERN.test(value)) {
    return 'Enter a valid email address.';
  }
  return null;
}

export function validatePasswordInput(password) {
  const value = String(password ?? '');
  if (!value) {
    return 'Password is required.';
  }
  if (value.length < 6) {
    return 'Password must be at least 6 characters.';
  }
  return null;
}

export function validateLoginFields({ email, password }) {
  return validateEmailInput(email) || validatePasswordInput(password);
}

export function validateRegisterFields({ email, password, confirmPassword }) {
  return (
    validateEmailInput(email)
    || validatePasswordInput(password)
    || (password !== confirmPassword ? 'Passwords do not match.' : null)
  );
}
