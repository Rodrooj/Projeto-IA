import { useEffect, useRef, useState } from 'react';
import '@mediapipe/camera_utils';
import '@mediapipe/holistic';
import type { Results } from '@mediapipe/holistic';

const CameraClass = (window as any).Camera;
const HolisticClass = (window as any).Holistic;
import * as tflite from '@tensorflow/tfjs-tflite';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import { Volume2, VolumeX, Video, VideoOff, Home } from 'lucide-react';
import { Link } from 'react-router-dom';
import { extractAndNormalizeSpatial, FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME } from '../lib/libras';

// URL Mock do modelo para o esqueleto, na vida real seria importado da public/
const MODEL_URL = '/models/libras_model.tflite';

/**
 * Componente principal `LibrasTranslator`
 * 
 * Gerencia a lógica do pipeline do Tradutor de Libras:
 * 1. Inicializa o Backend TFLite.
 * 2. Captura imagens da Webcam (via MediaPipe Camera).
 * 3. Identifica landmarks espaciais (via MediaPipe Holistic).
 * 4. Acumula os landmarks temporais em um Buffer (tamanho = 30 frames).
 * 5. Realiza a inferência com o modelo TFLite Híbrido CNN-1D + LSTM.
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
  const sequenceBuffer = useRef<number[][]>([]);
  const lastSpoken = useRef<string>('');
  const speakTimeout = useRef<number | null>(null);

  const [classMapping, setClassMapping] = useState<Record<number, string>>({});

  /**
   * Efeito disparado na inicialização do componente para carregar 
   * os requisitos do TensorFlow Lite e o modelo em memória local.
   */
  useEffect(() => {
    // Inicialização do TensorFlow Lite Backend e Modelo
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
   * Callback executado pelo MediaPipe a cada frame processado com sucesso.
   * 
   * @param results - O objeto contendo as posições tridimensionais do esqueleto e mãos.
   */
  const onResults = (results: Results) => {
    // Desenhar Landmarks (opcional, foco na estética minimalista)
    const canvasCtx = canvasRef.current?.getContext('2d');
    if (canvasCtx && canvasRef.current && videoRef.current) {
      canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      // Aqui poderíamos usar drawConnectors do mediapipe/drawing_utils
    }

    if (!results.poseLandmarks) {
      // Ignora frames sem detecção de pessoa
      return;
    }

    // Engenharia de atributos (Espacial)
    const features = extractAndNormalizeSpatial(results);
    
    // Adiciona ao Buffer Temporal
    sequenceBuffer.current.push(features);

    // Mantém o tamanho do buffer exatamente em FRAMES_PER_SEQUENCE (30 frames)
    if (sequenceBuffer.current.length > FRAMES_PER_SEQUENCE) {
      sequenceBuffer.current.shift();
    }

    // Inferência
    if (sequenceBuffer.current.length === FRAMES_PER_SEQUENCE && tfliteModel.current) {
      runInference(sequenceBuffer.current);
    }
  };

  /**
   * Executa a predição local utilizando a IA baseada em TensorFlow Lite.
   * 
   * @param sequence - Tensor contendo 30 frames temporais com 159 features matemáticas cada.
   */
  const runInference = (sequence: number[][]) => {
    if (!tfliteModel.current) return;
    
    try {
      tf.tidy(() => {
         // Formato esperado: [1, 30, 159]
        const inputTensor = tf.tensor3d([sequence], [1, FRAMES_PER_SEQUENCE, FEATURES_PER_FRAME], 'float32');
        
        // Predict
        const outputTensor = tfliteModel.current!.predict(inputTensor) as any;
        const predictionsArray = outputTensor.dataSync();
        
        // Obter Classe
        const maxConfidence = Math.max(...predictionsArray);
        const predictedClassIdx = predictionsArray.indexOf(maxConfidence);
        const predictedText = classMapping[predictedClassIdx] || "Desconhecido";
        
        console.log(`Predição: ${predictedText} (${maxConfidence.toFixed(2)})`, predictionsArray);

        setPrediction(predictedText);
        setConfidence(maxConfidence);
        
        // Mitigação de Mode Collapse (Confidence > 70%) para a voz
        if (maxConfidence > 0.7) {
          // Synthesis
          triggerVoice(predictedText);
        }
      });
    } catch (e) {
      console.error("Erro na inferência", e);
    }
  };

  /**
   * Sintetiza o texto em formato de áudio (Text-to-Speech) caso
   * o usuário mantenha o controle de voz ativado.
   * Implementa também mecanismo de cooldown (3 segundos) para 
   * evitar superposição (stuttering) da fala da mesma palavra seguidamente.
   * 
   * @param text - A palavra ou texto reconhecido na predição para sintetizar.
   */
  const triggerVoice = (text: string) => {
    if (!isVoiceEnabled) return;
    
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
   * Liga ou desliga a instância da Câmera integrada com o MediaPipe Holistic.
   * Recarrega a página ao desligar devido a restrições do cleanup padrão 
   * da biblioteca mediapipe/camera_utils.
   */
  const toggleCamera = () => {
    if (isCameraActive) {
      setIsCameraActive(false);
      window.location.reload(); // Hard reset por limitações do MediaPipe Camera
      return;
    }

    if (!videoRef.current) return;

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
