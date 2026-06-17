import { describe, it, expect } from 'vitest';
import { extractAndNormalizeSpatial } from './libras';
import type { Results, Landmark } from '@mediapipe/holistic';

describe('Módulo de Engenharia de Atributos de LIBRAS', () => {
  describe('extractAndNormalizeSpatial', () => {
    it('deve utilizar o centro dos ombros como origem (UN-01)', () => {
      // Mock de landmarks com 53 posições (apenas as que importam para o teste)
      const mockPoseLandmarks: Landmark[] = new Array(33).fill({ x: 0, y: 0, z: 0, visibility: 0 });
      
      // Ombro Esquerdo (11) e Ombro Direito (12)
      mockPoseLandmarks[11] = { x: 0.4, y: 0.5, z: 0, visibility: 1 };
      mockPoseLandmarks[12] = { x: 0.6, y: 0.5, z: 0, visibility: 1 };
      // Ponto médio esperado = (0.5, 0.5, 0)
      
      // Um landmark qualquer de teste (ex: polegar esquerdo, embora o modelo não pegue mãos daqui, pegaremos um da pose)
      // O landmark 20 (dedo indicador direito na pose)
      mockPoseLandmarks[20] = { x: 0.8, y: 0.9, z: 0.1, visibility: 1 };

      const mockResults: Results = {
        poseLandmarks: mockPoseLandmarks,
        poseBlendshapes: undefined,
        faceBlendshapes: undefined,
        image: {} as any,
      } as any;

      const normalized = extractAndNormalizeSpatial(mockResults);

      // O landmark 20 é o 10º elemento no array poseIndices [11, 12, 13, 14, 15, 16, 23, 24, 19, 20, 21]
      // Então as coordenadas (x,y,z) do índice 20 estarão na posição 9 * 3 = 27 (0-indexed base)
      const landmark20X = normalized[27];
      const landmark20Y = normalized[28];
      const landmark20Z = normalized[29];

      // Ponto médio = (0.5, 0.5, 0)
      // 0.8 - 0.5 = 0.3
      // 0.9 - 0.5 = 0.4
      // 0.1 - 0 = 0.1
      expect(landmark20X).toBeCloseTo(0.3);
      expect(landmark20Y).toBeCloseTo(0.4);
      expect(landmark20Z).toBeCloseTo(0.1);
    });

    it('deve retornar um array com 159 features (3*11 pose + 3*21 leftHand + 3*21 rightHand)', () => {
      const mockResults: Results = {
        poseLandmarks: undefined,
        image: {} as any,
      } as any;

      const normalized = extractAndNormalizeSpatial(mockResults);
      expect(normalized.length).toBe(159);
    });
  });

  describe('Buffer Circular (Conceitual)', () => {
    it('deve preservar a estrutura temporal de 159 atributos', () => {
      const buffer: number[][] = [];
      const frameFeatures = new Array(159).fill(0.5);

      for (let i = 0; i < 35; i++) {
        buffer.push(frameFeatures);

        if (buffer.length > 30) {
          buffer.shift();
        }
      }

      expect(buffer.length).toBe(30);
      expect(buffer[0].length).toBe(159);
    });
  });
});
