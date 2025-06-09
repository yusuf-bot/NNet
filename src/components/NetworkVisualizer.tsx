
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
          className={`w-8 h-8 rounded-full border-2 border-foreground transition-all duration-500 ${
            isActive 
              ? 'bg-foreground shadow-lg' 
              : 'bg-background'
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
          className={`w-8 h-8 rounded-full border-2 border-foreground transition-all duration-500 ${
            isActive 
              ? 'bg-foreground shadow-lg' 
              : 'bg-background'
          }`}
        />
      );
    }
    
    return neurons;
  };

  const renderConnections = () => {
    const connections = [];
    const layerSpacing = 200; // Distance between layers
    const neuronSpacing = 48; // Distance between neurons in a layer
    
    for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
      const fromLayerSize = Math.min(layers[layerIndex], MAX_VISIBLE_NEURONS);
      const toLayerSize = Math.min(layers[layerIndex + 1], MAX_VISIBLE_NEURONS);
      
      const isActive = animationStep > layerIndex && animationStep > layerIndex + 1;
      
      // Calculate positions for from and to neurons
      const fromNeurons = layers[layerIndex] > MAX_VISIBLE_NEURONS ? fromLayerSize - 1 : fromLayerSize;
      const toNeurons = layers[layerIndex + 1] > MAX_VISIBLE_NEURONS ? toLayerSize - 1 : toLayerSize;
      
      // Create connections between all visible neurons
      for (let fromNeuron = 0; fromNeuron < fromNeurons; fromNeuron++) {
        for (let toNeuron = 0; toNeuron < toNeurons; toNeuron++) {
          const fromY = (fromNeuron - (fromNeurons - 1) / 2) * neuronSpacing;
          const toY = (toNeuron - (toNeurons - 1) / 2) * neuronSpacing;
          
          connections.push(
            <line
              key={`${layerIndex}-${fromNeuron}-${toNeuron}`}
              x1={layerIndex * layerSpacing + 16} // 16 is half of neuron width
              y1={fromY}
              x2={(layerIndex + 1) * layerSpacing + 16}
              y2={toY}
              stroke="hsl(var(--foreground))"
              strokeWidth={isActive ? "2" : "1"}
              opacity={isActive ? "0.8" : "0.3"}
              className="transition-all duration-500"
            />
          );
        }
      }
      
      // Add connections for the last neuron if there are hidden neurons
      if (layers[layerIndex] > MAX_VISIBLE_NEURONS) {
        const lastFromY = (fromLayerSize - 1 - (fromLayerSize - 1) / 2) * neuronSpacing + neuronSpacing;
        for (let toNeuron = 0; toNeuron < toNeurons; toNeuron++) {
          const toY = (toNeuron - (toNeurons - 1) / 2) * neuronSpacing;
          connections.push(
            <line
              key={`${layerIndex}-last-${toNeuron}`}
              x1={layerIndex * layerSpacing + 16}
              y1={lastFromY}
              x2={(layerIndex + 1) * layerSpacing + 16}
              y2={toY}
              stroke="hsl(var(--foreground))"
              strokeWidth={isActive ? "2" : "1"}
              opacity={isActive ? "0.8" : "0.3"}
              className="transition-all duration-500"
            />
          );
        }
      }
      
      // Add connections to the last neuron if there are hidden neurons in the next layer
      if (layers[layerIndex + 1] > MAX_VISIBLE_NEURONS) {
        const lastToY = (toLayerSize - 1 - (toLayerSize - 1) / 2) * neuronSpacing + neuronSpacing;
        for (let fromNeuron = 0; fromNeuron < fromNeurons; fromNeuron++) {
          const fromY = (fromNeuron - (fromNeurons - 1) / 2) * neuronSpacing;
          connections.push(
            <line
              key={`${layerIndex}-${fromNeuron}-last`}
              x1={layerIndex * layerSpacing + 16}
              y1={fromY}
              x2={(layerIndex + 1) * layerSpacing + 16}
              y2={lastToY}
              stroke="hsl(var(--foreground))"
              strokeWidth={isActive ? "2" : "1"}
              opacity={isActive ? "0.8" : "0.3"}
              className="transition-all duration-500"
            />
          );
        }
      }
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
      
      <div className="relative bg-background rounded-lg border-2 border-border p-8 min-h-96">
        {isInferenceRunning && (
          <div className="absolute top-4 right-4 bg-foreground text-background px-3 py-1 rounded-full text-sm font-medium animate-pulse">
            Processing...
          </div>
        )}
        
        <div className="flex items-center justify-center h-full min-h-80 overflow-x-auto">
          <svg
            width={layers.length * 200}
            height="400"
            viewBox={`0 0 ${layers.length * 200} 400`}
            className="mx-auto"
          >
            {/* Render connections first (behind neurons) */}
            <g transform="translate(0, 200)">
              {renderConnections()}
            </g>
            
            {/* Render neurons */}
            {layers.map((layerSize, layerIndex) => {
              const visibleNeurons = Math.min(layerSize, MAX_VISIBLE_NEURONS);
              const showEllipsis = layerSize > MAX_VISIBLE_NEURONS;
              const actualVisibleCount = showEllipsis ? visibleNeurons - 1 : visibleNeurons;
              const neuronSpacing = 48;
              
              return (
                <g key={layerIndex} transform={`translate(${layerIndex * 200}, 200)`}>
                  {/* Layer Label */}
                  <text
                    x="16"
                    y="-120"
                    textAnchor="middle"
                    className="text-lg font-semibold fill-foreground"
                  >
                    {layerIndex === 0 ? 'Input' : 
                     layerIndex === layers.length - 1 ? 'Output' : 
                     `Hidden ${layerIndex}`}
                  </text>
                  <text
                    x="16"
                    y="-100"
                    textAnchor="middle"
                    className="text-sm fill-muted-foreground"
                  >
                    ({layerSize})
                  </text>
                  
                  {/* Neurons */}
                  {Array.from({ length: actualVisibleCount }).map((_, neuronIndex) => {
                    const isActive = animationStep > layerIndex;
                    const y = (neuronIndex - (actualVisibleCount - 1) / 2) * neuronSpacing;
                    
                    return (
                      <circle
                        key={neuronIndex}
                        cx="16"
                        cy={y}
                        r="16"
                        fill={isActive ? "hsl(var(--foreground))" : "hsl(var(--background))"}
                        stroke="hsl(var(--foreground))"
                        strokeWidth="2"
                        className="transition-all duration-500"
                      />
                    );
                  })}
                  
                  {/* Ellipsis */}
                  {showEllipsis && (
                    <>
                      <text
                        x="16"
                        y={(actualVisibleCount - (actualVisibleCount - 1) / 2) * neuronSpacing + 24}
                        textAnchor="middle"
                        className="text-xs fill-muted-foreground"
                      >
                        ({layerSize - actualVisibleCount} more...)
                      </text>
                      
                      {/* Last neuron */}
                      <circle
                        cx="16"
                        cy={(actualVisibleCount - (actualVisibleCount - 1) / 2) * neuronSpacing + 48}
                        r="16"
                        fill={animationStep > layerIndex ? "hsl(var(--foreground))" : "hsl(var(--background))"}
                        stroke="hsl(var(--foreground))"
                        strokeWidth="2"
                        className="transition-all duration-500"
                      />
                    </>
                  )}
                </g>
              );
            })}
          </svg>
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
