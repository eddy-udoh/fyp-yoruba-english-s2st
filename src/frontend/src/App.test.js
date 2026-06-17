import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the MedSpeak header', () => {
  render(<App />);
  expect(screen.getByText(/MedSpeak/i)).toBeInTheDocument();
});

test('shows both translation direction toggles', () => {
  render(<App />);
  expect(screen.getByText(/Yorùbá → English/i)).toBeInTheDocument();
  expect(screen.getByText(/English → Yorùbá/i)).toBeInTheDocument();
});
