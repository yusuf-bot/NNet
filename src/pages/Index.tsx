
import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Upload, Brain, Play, Download, FileText } from 'lucide-react';
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
  architecture: ModelArchitecture;
  training_config: TrainingConfig;
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

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <Brain className="w-12 h-12 text-foreground mr-4" />
            <h1 className="text-5xl font-bold text-foreground">
              Neural Network Studio
            </h1>
          </div>
          <p className="text-xl text-muted-foreground font-medium">Build, Train, and Visualize Neural Networks</p>
        </div>

        {/* Main Content */}
        {activeView === 'menu' && (
          <div className="max-w-4xl mx-auto">
            <div className="grid md:grid-cols-2 gap-8">
              {/* Train New Model */}
              <Card className="border-2 hover:border-foreground transition-all duration-300 hover:shadow-lg">
                <div className="p-8 text-center">
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
              </Card>

              {/* Load Existing Model */}
              <Card className="border-2 hover:border-foreground transition-all duration-300 hover:shadow-lg">
                <div className="p-8 text-center">
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
              </Card>
            </div>

            {/* Features */}
            <div className="mt-16 grid md:grid-cols-3 gap-8 text-center">
              <div className="p-6">
                <Play className="w-12 h-12 mx-auto mb-4 text-foreground" />
                <h4 className="text-lg font-semibold mb-2 text-foreground">Real-time Visualization</h4>
                <p className="text-muted-foreground leading-relaxed">Watch data flow through your network with smooth animations</p>
              </div>
              <div className="p-6">
                <FileText className="w-12 h-12 mx-auto mb-4 text-foreground" />
                <h4 className="text-lg font-semibold mb-2 text-foreground">Batch & Single Predictions</h4>
                <p className="text-muted-foreground leading-relaxed">Make predictions on CSV files or individual inputs</p>
              </div>
              <div className="p-6">
                <Download className="w-12 h-12 mx-auto mb-4 text-foreground" />
                <h4 className="text-lg font-semibold mb-2 text-foreground">Export Models</h4>
                <p className="text-muted-foreground leading-relaxed">Save and share your trained models as .nn files</p>
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
              <div></div>
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
