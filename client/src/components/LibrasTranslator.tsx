import { useEffect, useRef, useState } from 'react';
import '@mediapipe/camera_utils';
import '@mediapipe/holistic';
import type { Results } from '@mediapipe/holistic';

import * as tflite from '@tensorflow/tfjs-tflite';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { Volume2, VolumeX, Video, VideoOff, Home } from 'lucide-react';
import { Link } from 'react-router-dom';
import { extractAndNormalizeSpatial, FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME } from '../lib/libras';

// URL do modelo TFLite para inferência no navegador
const MODEL_URL = '/models/libras_model.tflite';

/**
 * Frequência de inferência: executa a predição a cada N frames processados pelo MediaPipe.
 * Com a câmera a ~30fps, INFERENCE_EVERY_N_FRAMES=5 resulta em ~6 inferências/segundo,
 * um equilíbrio entre responsividade e performance do navegador.
 */
const INFERENCE_EVERY_N_FRAMES = 5;

/**
 * Componente principal `LibrasTranslator`
 * 
 * Gerencia a lógica do pipeline do Tradutor de Libras:
 * 1. Inicializa o Backend TFLite.
 * 2. Captura imagens da Webcam (via MediaPipe Camera).
 * 3. Identifica landmarks espaciais (via MediaPipe Holistic).
 * 4. Acumula os landmarks temporais em um Buffer Circular (tamanho = 30 frames).
 * 5. Realiza a inferência throttled com o modelo TFLite Híbrido CNN-1D + LSTM.
 * 6. Dispara a síntese de voz (Web Speech API) caso a predição ultrapasse 70% de confiança.
 */
export default function LibrasTranslator() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [prediction, setPrediction] = useState<string>('');
  const [confidence, setConfidence] = useState<number>(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState<boolean>(true);

  // Instâncias mantidas no componente
  const tfliteModel = useRef<tflite.TFLiteModel | null>(null);
  
  /**
   * Buffer Circular para sequências temporais — O(1) por inserção
   * ao invés de O(n) do Array.shift() anterior.
   * O writeIndex avança circularmente e frameCount rastreia o preenchimento.
   */
  const sequenceBuffer = useRef<(number[] | null)[]>(new Array(FRAMES_PER_SEQUENCE).fill(null));
  const bufferWriteIndex = useRef(0);
  const bufferFrameCount = useRef(0);
  
  const lastSpoken = useRef<string>('');
  const speakTimeout = useRef<number | null>(null);
  const inferenceCounter = useRef(0);
  
  /**
   * Refs "espelho" para evitar closures obsoletas.
   * O MediaPipe captura onResults uma única vez; estes refs garantem
   * que as funções sempre acessem o valor mais recente do estado.
   */
  const isVoiceEnabledRef = useRef(isVoiceEnabled);
  const classMappingRef = useRef<Record<number, string>>({});
  
  /**
   * Instâncias de câmera e holistic armazenadas para cleanup programático
   * sem necessidade de window.location.reload().
   */
  const cameraRef = useRef<any>(null);
  const holisticRef = useRef<any>(null);

  const [classMapping, setClassMapping] = useState<Record<number, string>>({});

  // Sincroniza refs com o estado React a cada atualização
  useEffect(() => { isVoiceEnabledRef.current = isVoiceEnabled; }, [isVoiceEnabled]);
  useEffect(() => { classMappingRef.current = classMapping; }, [classMapping]);

  /**
   * Efeito disparado na inicialização do componente para carregar 
   * os requisitos do TensorFlow Lite e o modelo em memória local.
   */
  useEffect(() => {
    const initModel = async () => {
      try {
        // Para tfjs-tflite funcionar no navegador
        tflite.setWasmPath('/');
        // Fetch class mapping
        try {
          const res = await fetch('/models/class_mapping.json');
          if (res.ok) {
            const data = await res.json();
            setClassMapping(data);
          }
        } catch(e) {
          console.warn("Could not load class mapping", e);
        }

        const model = await tflite.loadTFLiteModel(MODEL_URL);
        console.log("Model loaded. Inputs:", model.inputs);
        tfliteModel.current = model;
        setIsModelLoaded(true);
      } catch (err) {
        console.error("Erro: Modelo TFLite falhou ao carregar.", err);
        setErrorMsg("Erro ao carregar o modelo de IA. A tradução não funcionará.");
        setIsModelLoaded(true); // Permite iniciar a câmera apenas no modo visual
      }
    };
    initModel();
  }, []);

  /**
   * Recupera o conteúdo do buffer circular em ordem temporal correta.
   * Necessário porque o buffer circular escreve em posições arbitrárias;
   * a leitura precisa reconstruir a sequência cronológica para a LSTM.
   */
  const getOrderedBuffer = (): number[][] => {
    const buf = sequenceBuffer.current;
    const result: number[][] = [];
    const oldestIdx = bufferWriteIndex.current % FRAMES_PER_SEQUENCE;
    for (let i = 0; i < FRAMES_PER_SEQUENCE; i++) {
      const idx = (oldestIdx + i) % FRAMES_PER_SEQUENCE;
      result.push(buf[idx] || new Array(FEATURES_PER_FRAME).fill(0));
    }
    return result;
  };

  /**
   * Sintetiza o texto em formato de áudio (Text-to-Speech) caso
   * o usuário mantenha o controle de voz ativado.
   * Implementa mecanismo de cooldown (3 segundos) para 
   * evitar superposição (stuttering) da fala.
   * 
   * Usa isVoiceEnabledRef para acessar o estado atual sem depender de closures.
   * 
   * @param text - A palavra ou texto reconhecido na predição para sintetizar.
   */
  const triggerVoice = (text: string) => {
    // Acessa ref ao invés do state para evitar closure obsoleta
    if (!isVoiceEnabledRef.current) return;
    
    // Evita repetição insana em curtos períodos
    if (lastSpoken.current === text) return;
    
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'pt-BR';
      utterance.rate = 1.0;
      window.speechSynthesis.speak(utterance);
      
      lastSpoken.current = text;
      
      if (speakTimeout.current) clearTimeout(speakTimeout.current);
      speakTimeout.current = window.setTimeout(() => {
        lastSpoken.current = '';
      }, 3000); // 3 segundos de cooldown
    }
  };

  /**
   * Executa a predição local utilizando a IA baseada em TensorFlow Lite.
   * 
   * Melhorias sobre a versão anterior:
   * - Usa `outputTensor.data()` (async) ao invés de `dataSync()` bloqueante.
   * - Usa argmax manual ao invés de `Math.max(...spread)` para evitar
   *   stack overflow com muitas classes.
   * - Libera tensores manualmente (tf.tidy não suporta funções async).
   * 
   * @param sequence - Array de 30 frames temporais com 159 features matemáticas cada.
   */
  const runInference = async (sequence: number[][]) => {
    if (!tfliteModel.current) return;
    
    let inputTensor: tf.Tensor | null = null;
    try {
      // Formato esperado: [1, 30, 159]
      inputTensor = tf.tensor3d([sequence], [1, FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME], 'float32');
      
      // Predict
      const outputTensor = tfliteModel.current.predict(inputTensor) as tf.Tensor;
      // Async data() ao invés de dataSync() — não bloqueia a thread principal
      const predictionsArray = await outputTensor.data();
      
      // Argmax manual — seguro para qualquer quantidade de classes,
      // ao invés de Math.max(...predictionsArray) que copia para a call stack
      let maxConf = -Infinity;
      let predIdx = 0;
      for (let i = 0; i < predictionsArray.length; i++) {
        if (predictionsArray[i] > maxConf) {
          maxConf = predictionsArray[i];
          predIdx = i;
        }
      }
      
      // Usa ref ao invés de state para evitar closure obsoleta
      const currentMapping = classMappingRef.current;
      const predictedText = currentMapping[predIdx] || "Desconhecido";
      
      console.log(`Predição: ${predictedText} (${maxConf.toFixed(2)})`);

      setPrediction(predictedText);
      setConfidence(maxConf);
      
      // Mitigação de Mode Collapse (Confidence > 70%) para a voz
      if (maxConf > 0.7) {
        triggerVoice(predictedText);
      }
      
      // Libera tensores manualmente (tf.tidy não suporta async)
      outputTensor.dispose();
    } catch (e) {
      console.error("Erro na inferência", e);
    } finally {
      inputTensor?.dispose();
    }
  };

  /**
   * Callback executado pelo MediaPipe a cada frame processado com sucesso.
   * 
   * Usa buffer circular para O(1) e inferência throttled para performance.
   * Todas as dependências são acessadas via refs, então esta função é segura
   * mesmo quando capturada pela instância do MediaPipe Holistic.
   * 
   * @param results - O objeto contendo as posições tridimensionais do esqueleto e mãos.
   */
  const onResults = (results: Results) => {
    // Desenhar Landmarks (opcional, foco na estética minimalista)
    const canvasCtx = canvasRef.current?.getContext('2d');
    if (canvasCtx && canvasRef.current) {
      canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    if (!results.poseLandmarks) {
      // Ignora frames sem detecção de pessoa
      return;
    }

    // Engenharia de atributos (Espacial)
    const features = extractAndNormalizeSpatial(results);
    
    // Inserção no buffer circular: O(1) ao invés de Array.shift() O(n)
    const writeIdx = bufferWriteIndex.current % FRAMES_PER_SEQUENCE;
    sequenceBuffer.current[writeIdx] = features;
    bufferWriteIndex.current++;
    bufferFrameCount.current = Math.min(bufferFrameCount.current + 1, FRAMES_PER_SEQUENCE);

    // Inferência throttled: executa apenas a cada N frames para não sobrecarregar o navegador
    inferenceCounter.current++;
    if (
      bufferFrameCount.current >= FRAMES_PER_SEQUENCE &&
      tfliteModel.current &&
      inferenceCounter.current % INFERENCE_EVERY_N_FRAMES === 0
    ) {
      const orderedSequence = getOrderedBuffer();
      runInference(orderedSequence);
    }
  };

  /**
   * Liga ou desliga a instância da Câmera integrada com o MediaPipe Holistic.
   * 
   * Substituído o `window.location.reload()` por cleanup programático:
   * - `camera.stop()` para parar a captura
   * - Revogação dos MediaStream tracks
   * - `holistic.close()` para liberar recursos do MediaPipe
   * Isso preserva o modelo TFLite carregado e o contexto do usuário.
   */
  const toggleCamera = () => {
    if (isCameraActive) {
      // Cleanup programático sem reload — preserva o modelo carregado
      try {
        cameraRef.current?.stop();
        // Revoga os MediaStream tracks do navegador
        const stream = videoRef.current?.srcObject as MediaStream | null;
        stream?.getTracks().forEach(track => track.stop());
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
        holisticRef.current?.close();
      } catch (e) {
        console.warn("Erro ao parar câmera/holistic:", e);
      }
      
      cameraRef.current = null;
      holisticRef.current = null;
      
      // Reset do buffer circular
      sequenceBuffer.current = new Array(FRAMES_PER_SEQUENCE).fill(null);
      bufferWriteIndex.current = 0;
      bufferFrameCount.current = 0;
      inferenceCounter.current = 0;
      
      setIsCameraActive(false);
      setPrediction('');
      setConfidence(0);
      return;
    }

    if (!videoRef.current) return;

    // Verifica se as bibliotecas MediaPipe estão disponíveis globalmente
    const CameraClass = (window as any).Camera;
    const HolisticClass = (window as any).Holistic;
    
    if (!CameraClass || !HolisticClass) {
      setErrorMsg("Erro: Bibliotecas do MediaPipe não carregaram. Verifique sua conexão com a internet.");
      console.error("MediaPipe Camera or Holistic not found on window object");
      return;
    }

    try {
      const holistic = new HolisticClass({
        locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
      });

      holistic.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        smoothSegmentation: true,
        refineFaceLandmarks: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      holistic.onResults(onResults);
      holisticRef.current = holistic;

      const camera = new CameraClass(videoRef.current, {
        onFrame: async () => {
          if (videoRef.current) {
            try {
              await holistic.send({ image: videoRef.current });
            } catch (e) {
              console.error("Erro no holistic.send", e);
            }
          }
        },
        width: 640,
        height: 360
      });

      camera.start();
      cameraRef.current = camera;
      setIsCameraActive(true);
      setErrorMsg(null);
    } catch (err) {
      setErrorMsg("Erro ao acessar a câmera. Verifique as permissões.");
      console.error(err);
    }
  };

  return (
    <div className="translator-container">
      <div className="video-section">
        {!isModelLoaded && (
          <div className="loading-overlay">
            <div className="loader"></div>
            <p>Carregando Inteligência Artificial Edge...</p>
          </div>
        )}
        
        <video 
          ref={videoRef} 
          className="video-element" 
          playsInline 
          autoPlay 
          muted
        />
        <canvas 
          ref={canvasRef} 
          className="canvas-overlay"
          width={640} 
          height={360} 
        />
        
        {errorMsg && (
          <div className="loading-overlay" style={{ background: 'rgba(220, 38, 38, 0.9)' }}>
            <p>{errorMsg}</p>
          </div>
        )}
      </div>

      <div className="panel-section">
        <div className="header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1>Libras AI</h1>
            <p>Tradução em tempo real no dispositivo</p>
          </div>
          <Link to="/" style={{ color: 'var(--text-muted)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <Home size={20} />
          </Link>
        </div>

        <div className={`result-card ${prediction ? 'active' : ''}`}>
          <div className="translation-text">
            {prediction || "..."}
          </div>
          
          <div className="confidence-bar">
            <div 
              className="confidence-level" 
              style={{ width: `${Math.round(confidence * 100)}%` }}
            ></div>
          </div>
          <p style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Confiança: {Math.round(confidence * 100)}%
          </p>
        </div>

        <div className="controls">
          <button 
            className="btn" 
            onClick={toggleCamera}
            disabled={!isModelLoaded}
            style={{ backgroundColor: isCameraActive ? '#dc2626' : 'var(--primary-color)' }}
          >
            {isCameraActive ? <VideoOff size={20} /> : <Video size={20} />}
            {isCameraActive ? 'Encerrar Câmera' : 'Iniciar Câmera'}
          </button>
          
          <button 
            className="btn" 
            onClick={() => setIsVoiceEnabled(!isVoiceEnabled)}
            style={{ 
              backgroundColor: isVoiceEnabled ? 'var(--primary-color)' : 'transparent',
              border: isVoiceEnabled ? 'none' : '1px solid var(--border-color)',
              color: isVoiceEnabled ? 'white' : 'var(--text-primary)',
              marginTop: '0.5rem'
            }}
          >
            {isVoiceEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
            {isVoiceEnabled ? 'Voz Ativada' : 'Voz Desativada'}
          </button>
        </div>
      </div>
    </div>
  );
}
