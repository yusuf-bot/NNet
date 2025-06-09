import React, { useEffect, useState } from 'react';

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
          className={`w-8 h-8 rounded-full border-2 border-black transition-all duration-500 ${
            isActive 
              ? 'bg-black shadow-lg' 
              : 'bg-white'
          }`}
        />
      );
    }
    
    // Add ellipsis if needed
    if (showEllipsis) {
      neurons.push(
        <div key="ellipsis" className="text-xs text-gray-600 text-center py-2">
          ({layerSize - actualVisibleCount} more...)
        </div>
      );
      
      // Add last neuron
      const isActive = animationStep > layerIndex;
      neurons.push(
        <div
          key="last"
          className={`w-8 h-8 rounded-full border-2 border-black transition-all duration-500 ${
            isActive 
              ? 'bg-black shadow-lg' 
              : 'bg-white'
          }`}
        />
      );
    }
    
    return neurons;
  };

  const renderConnections = () => {
    const connections = [];
    const layerSpacing = 200;
    const neuronSpacing = 48;
    
    for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
      const fromLayerSize = Math.min(layers[layerIndex], MAX_VISIBLE_NEURONS);
      const toLayerSize = Math.min(layers[layerIndex + 1], MAX_VISIBLE_NEURONS);
      
      const isActive = animationStep > layerIndex && animationStep > layerIndex + 1;
      
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
              x1={layerIndex * layerSpacing + 16}
              y1={fromY}
              x2={(layerIndex + 1) * layerSpacing + 16}
              y2={toY}
              stroke="black"
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
              stroke="black"
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
              stroke="black"
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
    <div className="p-6 bg-white">
      {isInferenceRunning && (
        <div className="absolute top-4 right-4 bg-black text-white px-3 py-1 rounded-full text-sm font-medium animate-pulse">
          Processing...
        </div>
      )}
      
      <div className="flex items-center justify-center h-full min-h-80 overflow-x-auto">
        <div className="relative">
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
                        fill={isActive ? "black" : "white"}
                        stroke="black"
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
                        className="text-xs fill-gray-600"
                      >
                        ({layerSize - actualVisibleCount} more...)
                      </text>
                      
                      {/* Last neuron */}
                      <circle
                        cx="16"
                        cy={(actualVisibleCount - (actualVisibleCount - 1) / 2) * neuronSpacing + 48}
                        r="16"
                        fill={animationStep > layerIndex ? "black" : "white"}
                        stroke="black"
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
      
      <div className="mt-4 text-center text-gray-600 text-sm">
        {isInferenceRunning ? 
          'Watch the data flow through your neural network!' : 
          'Make a prediction to see the network in action'
        }
      </div>
    </div>
  );
};

export default NetworkVisualizer;
