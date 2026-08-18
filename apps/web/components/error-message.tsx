export function ErrorMessage({ message }: { message: string | null }) {
  return message ? <p className="error" role="alert">{message}</p> : null;
}

