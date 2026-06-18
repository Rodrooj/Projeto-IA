import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import LibrasTranslator from './LibrasTranslator';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import * as tflite from '@tensorflow/tfjs-tflite';

// Mock TFLite
vi.mock('@tensorflow/tfjs-tflite', () => {
  return {
    setWasmPath: vi.fn(),
    loadTFLiteModel: vi.fn(),
  };
});

/**
 * Helper para renderizar o componente dentro do contexto do Router,
 * necessário porque LibrasTranslator usa `<Link>` do react-router-dom.
 */
function renderWithRouter(ui: React.ReactElement) {
  return render(<MemoryRouter>{ui}</MemoryRouter>);
}

describe('LibrasTranslator Component', () => {
  let mockPredict: any;

  beforeEach(() => {
    vi.clearAllMocks();
    
    // Mock global fetch para evitar requisições HTTP reais no happy-dom
    vi.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    } as Response);
    
    // Configura o mock do modelo TFLite
    // Usa data() (async) ao invés de dataSync() para refletir a otimização aplicada
    mockPredict = vi.fn().mockReturnValue({
      data: () => Promise.resolve(new Float32Array([0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
      dispose: vi.fn(),
    });

    (tflite.loadTFLiteModel as any).mockResolvedValue({
      predict: mockPredict,
      inputs: [],
    });

    // Mock MediaPipe globals (normalmente injetados via CDN)
    (window as any).Holistic = vi.fn().mockImplementation(() => ({
      setOptions: vi.fn(),
      onResults: vi.fn(),
      send: vi.fn().mockResolvedValue(undefined),
      close: vi.fn(),
    }));
    (window as any).Camera = vi.fn().mockImplementation(() => ({
      start: vi.fn(),
      stop: vi.fn(),
    }));
  });

  it('deve desabilitar o botão Iniciar Câmera enquanto o modelo não carrega', () => {
    // Faremos o mock demorar para simular o carregamento
    (tflite.loadTFLiteModel as any).mockImplementation(() => new Promise(() => {}));
    
    renderWithRouter(<LibrasTranslator />);
    
    const button = screen.getByRole('button', { name: /iniciar câmera/i });
    expect(button).toBeDisabled();
    expect(screen.getByText(/carregando inteligência artificial/i)).toBeInTheDocument();
  });

  it('deve habilitar o botão Iniciar Câmera após carregar o modelo', async () => {
    renderWithRouter(<LibrasTranslator />);
    
    const button = await screen.findByRole('button', { name: /iniciar câmera/i });
    expect(button).not.toBeDisabled();
    expect(screen.queryByText(/carregando inteligência artificial/i)).not.toBeInTheDocument();
  });

  it('deve permitir clicar no botão de câmera sem gerar erro após modelo carregado', async () => {
    renderWithRouter(<LibrasTranslator />);
    
    const button = await screen.findByRole('button', { name: /iniciar câmera/i });
    
    // Clica no botão — em happy-dom o videoRef.current é null então toggleCamera
    // retorna cedo sem erro (guard clause). Verificamos que nenhum estado de erro
    // é exibido e o botão continua interativo.
    fireEvent.click(button);

    // Não deve exibir mensagem de erro de MediaPipe (as bibliotecas foram mockadas no window)
    expect(screen.queryByText(/erro.*mediapipe/i)).not.toBeInTheDocument();
    
    // O botão continua disponível para interação
    expect(screen.getByRole('button', { name: /iniciar câmera/i })).not.toBeDisabled();
    
    // NOTA: Teste end-to-end da câmera ativa (videoRef conectado, Holistic processando)
    // é responsabilidade do Playwright, não do vitest com happy-dom.
  });
});
