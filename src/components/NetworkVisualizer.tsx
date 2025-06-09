
import React, { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';

interface ModelData {
  architecture: {
    input_size: number;
    output_size: number;
    hidden_layers: number[];
  };
  training_config: {
    activation_names: string[];
    loss_function: string;
    learning_rate: number;
  };
}

interface NetworkVisualizerProps {
  modelData: ModelData;
  isInferenceRunning: boolean;
}

const MAX_VISIBLE_NEURONS = 8;

const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({ modelData, isInferenceRunning }) => {
  const [animationStep, setAnimationStep] = useState(0);

  const { input_size, output_size, hidden_layers } = modelData.architecture;
  const layers = [input_size, ...hidden_layers, output_size];

  // Animation logic
  useEffect(() => {
    if (!isInferenceRunning) {
      setAnimationStep(0);
      return;
    }

    const animationInterval = setInterval(() => {
      setAnimationStep(prev => {
        const maxLayers = layers.length;
        if (prev >= maxLayers) {
          return 0; // Reset animation
        }
        return prev + 1;
      });
    }, 800);

    return () => clearInterval(animationInterval);
  }, [isInferenceRunning, layers.length]);

  const renderLayer = (layerSize: number, layerIndex: number) => {
    const visibleNeurons = Math.min(layerSize, MAX_VISIBLE_NEURONS);
    const showEllipsis = layerSize > MAX_VISIBLE_NEURONS;
    const actualVisibleCount = showEllipsis ? visibleNeurons - 1 : visibleNeurons;
    
    const neurons = [];
    
    // Add visible neurons
    for (let i = 0; i < actualVisibleCount; i++) {
      const isActive = animationStep > layerIndex;
      neurons.push(
        <div
          key={i}
          className={`w-8 h-8 rounded-full border-2 transition-all duration-500 ${
            isActive 
              ? 'bg-foreground border-foreground shadow-lg' 
              : 'bg-background border-border'
          }`}
        />
      );
    }
    
    // Add ellipsis if needed
    if (showEllipsis) {
      neurons.push(
        <div key="ellipsis" className="text-xs text-muted-foreground text-center py-2">
          ({layerSize - actualVisibleCount} more...)
        </div>
      );
      
      // Add last neuron
      const isActive = animationStep > layerIndex;
      neurons.push(
        <div
          key="last"
          className={`w-8 h-8 rounded-full border-2 transition-all duration-500 ${
            isActive 
              ? 'bg-foreground border-foreground shadow-lg' 
              : 'bg-background border-border'
          }`}
        />
      );
    }
    
    return neurons;
  };

  const renderConnections = () => {
    const connections = [];
    
    for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
      const fromLayerSize = Math.min(layers[layerIndex], MAX_VISIBLE_NEURONS);
      const toLayerSize = Math.min(layers[layerIndex + 1], MAX_VISIBLE_NEURONS);
      
      const isActive = animationStep > layerIndex && animationStep > layerIndex + 1;
      
      // Create a visual representation of connections between layers
      connections.push(
        <div key={layerIndex} className="flex-1 flex items-center justify-center">
          <div className={`h-px bg-border transition-all duration-500 w-full ${
            isActive ? 'bg-foreground opacity-60' : 'opacity-20'
          }`} />
        </div>
      );
    }
    
    return connections;
  };

  return (
    <div className="border-2 border-border rounded-lg p-6 bg-background">
      <div className="mb-6">
        <h3 className="text-xl font-semibold text-foreground mb-2">Network Architecture</h3>
        <div className="flex items-center space-x-4 text-sm text-muted-foreground">
          <span>Input: {modelData.architecture.input_size}</span>
          <span>Hidden: [{modelData.architecture.hidden_layers.join(', ')}]</span>
          <span>Output: {modelData.architecture.output_size}</span>
          <span>Loss: {modelData.training_config.loss_function.toUpperCase()}</span>
        </div>
      </div>
      
      <div className="relative bg-white rounded-lg border-2 border-border p-8 min-h-96">
        {isInferenceRunning && (
          <div className="absolute top-4 right-4 bg-foreground text-background px-3 py-1 rounded-full text-sm font-medium animate-pulse">
            Processing...
          </div>
        )}
        
        <div className="flex items-center justify-between h-full min-h-80">
          {layers.map((layerSize, layerIndex) => (
            <React.Fragment key={layerIndex}>
              <div className="flex flex-col items-center space-y-4">
                {/* Layer Label */}
                <div className="text-center mb-4">
                  <div className="text-lg font-semibold text-foreground">
                    {layerIndex === 0 ? 'Input' : 
                     layerIndex === layers.length - 1 ? 'Output' : 
                     `Hidden ${layerIndex}`}
                  </div>
                  <div className="text-sm text-muted-foreground">({layerSize})</div>
                </div>
                
                {/* Neurons */}
                <div className="flex flex-col items-center space-y-3">
                  {renderLayer(layerSize, layerIndex)}
                </div>
              </div>
              
              {/* Connections */}
              {layerIndex < layers.length - 1 && (
                <div className="flex-1 mx-4">
                  {renderConnections()}
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
      
      <div className="mt-4 text-center text-muted-foreground text-sm">
        {isInferenceRunning ? 
          'Watch the data flow through your neural network!' : 
          'Make a prediction to see the network in action'
        }
      </div>
    </div>
  );
};

export default NetworkVisualizer;
