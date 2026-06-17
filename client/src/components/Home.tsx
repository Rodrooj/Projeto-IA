import { Link } from 'react-router-dom';
import { Sparkles, ArrowRight, ShieldCheck, Zap } from 'lucide-react';
import './Home.css';

export default function Home() {
  return (
    <div className="home-container">
      <header className="home-header">
        <div className="logo-container">
          <Sparkles className="logo-icon" />
          <h1>Libras AI</h1>
        </div>
      </header>

      <main className="home-main">
        <section className="hero-section">
          <h2>Tradução em Tempo Real, Diretamente no seu Dispositivo</h2>
          <p className="hero-subtitle">
            Uma solução inovadora que utiliza Inteligência Artificial Edge para traduzir a Língua Brasileira de Sinais (LIBRAS) com alta precisão e privacidade.
          </p>
          
          <Link to="/translator" className="cta-button">
            <span>Acessar Tradutor</span>
            <ArrowRight size={20} />
          </Link>
        </section>

        <section className="features-section">
          <div className="feature-card">
            <div className="feature-icon"><Zap size={24} /></div>
            <h3>Baixa Latência</h3>
            <p>Processamento rápido utilizando arquitetura híbrida CNN-1D + LSTM executada localmente.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon"><ShieldCheck size={24} /></div>
            <h3>Privacidade Total</h3>
            <p>Todo o processamento acontece no seu navegador. Nenhuma imagem é enviada para servidores externos.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon"><Sparkles size={24} /></div>
            <h3>Alta Precisão</h3>
            <p>Rastreamento detalhado das mãos, face e corpo usando MediaPipe Holistic e Engenharia de Atributos avançada.</p>
          </div>
        </section>
      </main>

      <footer className="home-footer">
        <p>Projeto Desenvolvido para Tradução Inclusiva de LIBRAS.</p>
      </footer>
    </div>
  );
}
