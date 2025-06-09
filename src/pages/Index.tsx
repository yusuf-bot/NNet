
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <Brain className="w-12 h-12 text-purple-400 mr-4" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Neural Network Studio
            </h1>
          </div>
          <p className="text-xl text-slate-300">Build, Train, and Visualize Neural Networks</p>
        </div>

        {/* Main Content */}
        {activeView === 'menu' && (
          <div className="max-w-4xl mx-auto">
            <div className="grid md:grid-cols-2 gap-8">
              {/* Train New Model */}
              <Card className="bg-slate-800/50 border-purple-500/20 backdrop-blur-sm hover:bg-slate-800/70 transition-all duration-300 hover:scale-105">
                <div className="p-8 text-center">
                  <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                    <Brain className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold mb-4 text-purple-300">Train New Model</h3>
                  <p className="text-slate-300 mb-6">
                    Upload your CSV data and configure a neural network from scratch
                  </p>
                  <Button 
                    onClick={() => setActiveView('train')}
                    className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold py-3"
                  >
                    Start Training
                  </Button>
                </div>
              </Card>

              {/* Load Existing Model */}
              <Card className="bg-slate-800/50 border-blue-500/20 backdrop-blur-sm hover:bg-slate-800/70 transition-all duration-300 hover:scale-105">
                <div className="p-8 text-center">
                  <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full flex items-center justify-center">
                    <Upload className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold mb-4 text-blue-300">Load Existing Model</h3>
                  <p className="text-slate-300 mb-6">
                    Upload a pre-trained .nn model file and start making predictions
                  </p>
                  <Button 
                    onClick={() => setActiveView('load')}
                    className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-semibold py-3"
                  >
                    Load Model
                  </Button>
                </div>
              </Card>
            </div>

            {/* Features */}
            <div className="mt-16 grid md:grid-cols-3 gap-6 text-center">
              <div className="p-6">
                <Play className="w-12 h-12 mx-auto mb-4 text-green-400" />
                <h4 className="text-lg font-semibold mb-2">Real-time Visualization</h4>
                <p className="text-slate-400">Watch data flow through your network with smooth animations</p>
              </div>
              <div className="p-6">
                <FileText className="w-12 h-12 mx-auto mb-4 text-yellow-400" />
                <h4 className="text-lg font-semibold mb-2">Batch & Single Predictions</h4>
                <p className="text-slate-400">Make predictions on CSV files or individual inputs</p>
              </div>
              <div className="p-6">
                <Download className="w-12 h-12 mx-auto mb-4 text-pink-400" />
                <h4 className="text-lg font-semibold mb-2">Export Models</h4>
                <p className="text-slate-400">Save and share your trained models as .nn files</p>
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
                className="border-slate-600 text-slate-300 hover:bg-slate-700"
              >
                ← Back to Menu
              </Button>
              <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
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
