import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Upload, Brain, Download } from 'lucide-react';
import TrainingPanel from '@/components/TrainingPanel';
import NetworkVisualizer from '@/components/NetworkVisualizer';
import PredictionPanel from '@/components/PredictionPanel';
import ModelLoader from '@/components/ModelLoader';

interface ModelArchitecture {
  input_size: number;
  output_size: number;
  hidden_layers: number[];
}

interface TrainingConfig {
  activation_names: string[];
  loss_function: string;
  learning_rate: number;
}

interface ModelData {
  model_info: {
    created_at: string;
    version: string;
  };
  architecture: ModelArchitecture;
  training_config: TrainingConfig;
  layers: any[];
  metadata: any;
}

const Index = () => {
  const [activeView, setActiveView] = useState<'menu' | 'train' | 'load' | 'visualize'>('menu');
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [isInferenceRunning, setIsInferenceRunning] = useState(false);

  const handleModelTrained = (data: ModelData) => {
    setModelData(data);
    setActiveView('visualize');
  };

  const handleModelLoaded = (data: ModelData) => {
    setModelData(data);
    setActiveView('visualize');
  };

  const handleBackToMenu = () => {
    setActiveView('menu');
    setModelData(null);
  };

  const handleDownloadModel = () => {
    if (!modelData) return;
    
    const dataStr = JSON.stringify(modelData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `model_${modelData.model_info.created_at.replace(/[:.]/g, '-')}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <Brain className="w-12 h-12 text-foreground mr-4" />
            <h1 className="text-5xl font-bold text-foreground">
              BUILD-A-BRAIN
            </h1>
          </div>
          <p className="text-xl text-muted-foreground font-medium">Build, Train, and Visualize Neural Networks</p>
        </div>

        {/* Main Content */}
        {activeView === 'menu' && (
          <div className="max-w-3xl mx-auto">
            <div className="text-center max-w-3xl">
    <p className="text-lg mb-4">
      A lightweight, intuitive tool to build and train neural networks on any CSV dataset — no coding required.
    </p>
    <p className="mb-4">
      Originally this started as a personal project, this app was designed to make it easy to experiment with neural networks using just <strong>NumPy</strong> and <strong>Pandas</strong>. Whether you're working with labeled or unlabeled data, the tool helps you quickly configure models, train them, and visualize performance — all from a simple interface.
    </p>
    <p>
      Perfect for students, hobbyists, and anyone curious about how neural networks work under the hood.
    </p>
    <div className="mt-6 space-x-6">
      <p> Check out my other projects at <a href="https://yusuf-bot.github.io" className="text-blue-600 underline hover:text-blue-800" target="_blank" rel="noopener noreferrer">
        My Website
      </a></p>
      <p> Check out the source code for this at <a href="https://github.com/yusuf-bot/build-a-brain" className="text-blue-600 underline hover:text-blue-800" target="_blank" rel="noopener noreferrer">
        Build-A-Brain on GitHub
      </a></p>
    </div>
  </div>
            <div className="grid md:grid-cols-2 gap-8">
              {/* Train New Model */}
              <div className="p-8 text-center border-2 border-border rounded-lg hover:border-foreground transition-colors">
                <div className="w-16 h-16 mx-auto mb-6 bg-foreground rounded-full flex items-center justify-center">
                  <Brain className="w-8 h-8 text-background" />
                </div>
                <h3 className="text-2xl font-semibold mb-4 text-foreground">Train New Model</h3>
                <p className="text-muted-foreground mb-6 leading-relaxed">
                  Upload your CSV data and configure a neural network from scratch
                </p>
                <Button 
                  onClick={() => setActiveView('train')}
                  className="w-full bg-foreground hover:bg-foreground/90 text-background font-medium py-3"
                >
                  Start Training
                </Button>
              </div>

              {/* Load Existing Model */}
              <div className="p-8 text-center border-2 border-border rounded-lg hover:border-foreground transition-colors">
                <div className="w-16 h-16 mx-auto mb-6 bg-foreground rounded-full flex items-center justify-center">
                  <Upload className="w-8 h-8 text-background" />
                </div>
                <h3 className="text-2xl font-semibold mb-4 text-foreground">Load Existing Model</h3>
                <p className="text-muted-foreground mb-6 leading-relaxed">
                  Upload a pre-trained .nn model file and start making predictions
                </p>
                <Button 
                  onClick={() => setActiveView('load')}
                  className="w-full bg-foreground hover:bg-foreground/90 text-background font-medium py-3"
                >
                  Load Model
                </Button>
              </div>
            </div>
          </div>
        )}

        {activeView === 'train' && (
          <TrainingPanel 
            onModelTrained={handleModelTrained}
            onBack={handleBackToMenu}
          />
        )}

        {activeView === 'load' && (
          <ModelLoader 
            onModelLoaded={handleModelLoaded}
            onBack={handleBackToMenu}
          />
        )}

        {activeView === 'visualize' && modelData && (
          <div className="space-y-8">
            <div className="flex items-center justify-between">
              <Button 
                onClick={handleBackToMenu}
                variant="outline"
                className="border-2 border-foreground text-foreground hover:bg-foreground hover:text-background"
              >
                ← Back to Menu
              </Button>
              <h2 className="text-3xl font-bold text-foreground">
                Neural Network Visualizer
              </h2>
              <Button 
                onClick={handleDownloadModel}
                className="bg-foreground hover:bg-foreground/90 text-background font-medium"
              >
                <Download className="w-4 h-4 mr-2" />
                Download Model
              </Button>
            </div>
            
            <div className="grid lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2">
                <NetworkVisualizer 
                  modelData={modelData}
                  isInferenceRunning={isInferenceRunning}
                />
              </div>
              <div>
                <PredictionPanel 
                  modelData={modelData}
                  onInferenceStart={() => setIsInferenceRunning(true)}
                  onInferenceEnd={() => setIsInferenceRunning(false)}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Index;