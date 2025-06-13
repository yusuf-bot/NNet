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

  // Get layer names
  const getLayerName = (layerIndex: number) => {
    if (layerIndex === 0) return 'Input Layer';
    if (layerIndex === layers.length - 1) return 'Output Layer';
    return `Hidden Layer ${layerIndex}`;
  };

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
              x1={layerIndex * layerSpacing + leftPadding + 16}
              y1={fromY}
              x2={(layerIndex + 1) * layerSpacing + leftPadding + 16}
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
              x1={layerIndex * layerSpacing + leftPadding + 16}
              y1={lastFromY}
              x2={(layerIndex + 1) * layerSpacing + leftPadding + 16}
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
              x1={layerIndex * layerSpacing + leftPadding + 16}
              y1={fromY}
              x2={(layerIndex + 1) * layerSpacing + leftPadding + 16}
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

  // Calculate dimensions for proper sizing
  const layerSpacing = 200;
  const neuronSpacing = 48;
  const maxNeuronsInLayer = Math.max(...layers.map(size => Math.min(size, MAX_VISIBLE_NEURONS)));
  const svgWidth = layers.length * layerSpacing + 100; // Extra padding for titles
  const svgHeight = Math.max(500, maxNeuronsInLayer * neuronSpacing + 300); // Extra space for titles and ellipsis
  const centerY = svgHeight / 2;
  const leftPadding = 50; // Padding to prevent title cutoff

  return (
    <div className="p-6 bg-white">
      {isInferenceRunning && (
        <div className="absolute top-4 right-4 bg-black text-white px-3 py-1 rounded-full text-sm font-medium animate-pulse">
          Processing...
        </div>
      )}
      
      <div className="flex items-center justify-center w-full overflow-x-auto">
        <div className="min-w-fit">
          <svg
            width={svgWidth}
            height={svgHeight}
            viewBox={`0 0 ${svgWidth} ${svgHeight}`}
            className="mx-auto"
            style={{ minWidth: svgWidth }}
          >
            {/* Render connections first (behind neurons) */}
            <g transform={`translate(0, ${centerY})`}>
              {renderConnections()}
            </g>
            
            {/* Render layer titles and neurons */}
            {layers.map((layerSize, layerIndex) => {
              const visibleNeurons = Math.min(layerSize, MAX_VISIBLE_NEURONS);
              const showEllipsis = layerSize > MAX_VISIBLE_NEURONS;
              const actualVisibleCount = showEllipsis ? visibleNeurons - 1 : visibleNeurons;
              
              return (
                <g key={layerIndex}>
                  {/* Layer Title */}
                  <text
                    x={layerIndex * layerSpacing + leftPadding + 16}
                    y={60}
                    textAnchor="middle"
                    className="text-sm font-semibold fill-black"
                  >
                    {getLayerName(layerIndex)}
                  </text>
                  <text
                    x={layerIndex * layerSpacing + leftPadding + 16}
                    y={78}
                    textAnchor="middle"
                    className="text-xs fill-gray-600"
                  >
                    ({layerSize} neurons)
                  </text>
                  
                  {/* Neurons */}
                  <g transform={`translate(${layerIndex * layerSpacing + leftPadding}, ${centerY})`}>
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
                          style={{
                            filter: isActive ? 'drop-shadow(0 0 6px rgba(0,0,0,0.3))' : 'none'
                          }}
                        />
                      );
                    })}
                    
                    {/* Ellipsis and last neuron */}
                    {showEllipsis && (
                      <>
                        {/* Last neuron */}
                        <circle
                          cx="16"
                          cy={(actualVisibleCount - (actualVisibleCount - 1) / 2) * neuronSpacing + 48}
                          r="16"
                          fill={animationStep > layerIndex ? "black" : "white"}
                          stroke="black"
                          strokeWidth="2"
                          className="transition-all duration-500"
                          style={{
                            filter: animationStep > layerIndex ? 'drop-shadow(0 0 6px rgba(0,0,0,0.3))' : 'none'
                          }}
                        />
                        
                        {/* Ellipsis text below the last neuron */}
                        <text
                          x="16"
                          y={(actualVisibleCount - (actualVisibleCount - 1) / 2) * neuronSpacing + 78}
                          textAnchor="middle"
                          className="text-xs fill-gray-600"
                        >
                          ...{layerSize - actualVisibleCount} more
                        </text>
                      </>
                    )}
                  </g>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
      
      {/* Network Architecture Summary */}
      <div className="mt-6 text-center text-sm text-gray-600">
        <p>Architecture: {layers.join(' → ')} neurons</p>
        {modelData.training_config.activation_names.length > 0 && (
          <p>Activations: {modelData.training_config.activation_names.join(', ')}</p>
        )}
      </div>
    </div>
  );
};

export default NetworkVisualizer;