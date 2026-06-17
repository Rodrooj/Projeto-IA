import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock do window.speechSynthesis
Object.defineProperty(window, 'speechSynthesis', {
  value: {
    speak: vi.fn(),
    cancel: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    getVoices: vi.fn().mockReturnValue([]),
  },
  writable: true,
});

// Mock de classes globais que não existem no happy-dom
class SpeechSynthesisUtteranceMock {
  text: string;
  lang: string;
  rate: number;
  constructor(text: string) {
    this.text = text;
    this.lang = '';
    this.rate = 1;
  }
}
(window as any).SpeechSynthesisUtterance = SpeechSynthesisUtteranceMock;

// Mock das classes do MediaPipe e Canvas (são apenas classes vazias)
(window as any).Camera = class {
  start() {}
  stop() {}
};

(window as any).Holistic = class {
  setOptions() {}
  onResults() {}
  send() {}
};

// Precisamos também fazer um mock de createObjectURL
(window.URL as any).createObjectURL = vi.fn();
