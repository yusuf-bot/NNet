
import React, { useEffect, useRef, useState } from 'react';
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

interface Neuron {
  x: number;
  y: number;
  layer: number;
  index: number;
  activation: number;
  isActive: boolean;
}

interface Connection {
  from: Neuron;
  to: Neuron;
  weight: number;
  isActive: boolean;
}

const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({ modelData, isInferenceRunning }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [neurons, setNeurons] = useState<Neuron[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [animationStep, setAnimationStep] = useState(0);

  // Initialize network structure
  useEffect(() => {
    if (!modelData) return;

    const { input_size, output_size, hidden_layers } = modelData.architecture;
    const layers = [input_size, ...hidden_layers, output_size];
    
    const newNeurons: Neuron[] = [];
    const newConnections: Connection[] = [];
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const width = canvas.width;
    const height = canvas.height;
    const layerSpacing = width / (layers.length + 1);
    
    // Create neurons
    layers.forEach((layerSize, layerIndex) => {
      const neuronSpacing = height / (layerSize + 1);
      
      for (let i = 0; i < layerSize; i++) {
        newNeurons.push({
          x: layerSpacing * (layerIndex + 1),
          y: neuronSpacing * (i + 1),
          layer: layerIndex,
          index: i,
          activation: 0,
          isActive: false
        });
      }
    });
    
    // Create connections
    for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
      const currentLayerNeurons = newNeurons.filter(n => n.layer === layerIndex);
      const nextLayerNeurons = newNeurons.filter(n => n.layer === layerIndex + 1);
      
      currentLayerNeurons.forEach(fromNeuron => {
        nextLayerNeurons.forEach(toNeuron => {
          newConnections.push({
            from: fromNeuron,
            to: toNeuron,
            weight: Math.random() * 2 - 1, // Random weight between -1 and 1
            isActive: false
          });
        });
      });
    }
    
    setNeurons(newNeurons);
    setConnections(newConnections);
  }, [modelData]);

  // Animation loop
  useEffect(() => {
    if (!isInferenceRunning) {
      setAnimationStep(0);
      // Reset all activations
      setNeurons(prev => prev.map(n => ({ ...n, activation: 0, isActive: false })));
      setConnections(prev => prev.map(c => ({ ...c, isActive: false })));
      return;
    }

    const animationInterval = setInterval(() => {
      setAnimationStep(prev => {
        const maxLayers = modelData.architecture.hidden_layers.length + 2;
        if (prev >= maxLayers) {
          return maxLayers; // Stay at final state
        }
        return prev + 1;
      });
    }, 800);

    return () => clearInterval(animationInterval);
  }, [isInferenceRunning, modelData]);

  // Update neuron activations based on animation step
  useEffect(() => {
    if (animationStep === 0) return;

    setNeurons(prev => prev.map(neuron => ({
      ...neuron,
      isActive: neuron.layer < animationStep,
      activation: neuron.layer < animationStep ? Math.random() * 0.8 + 0.2 : 0
    })));

    setConnections(prev => prev.map(connection => ({
      ...connection,
      isActive: connection.from.layer < animationStep - 1 && connection.to.layer < animationStep
    })));
  }, [animationStep]);

  // Draw the network
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw connections
    connections.forEach(connection => {
      const { from, to, weight, isActive } = connection;
      
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      
      if (isActive) {
        ctx.strokeStyle = weight > 0 ? '#22d3ee' : '#f87171';
        ctx.lineWidth = Math.abs(weight) * 3 + 1;
        ctx.globalAlpha = 0.8;
      } else {
        ctx.strokeStyle = '#374151';
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.3;
      }
      
      ctx.stroke();
      ctx.globalAlpha = 1;
    });

    // Draw neurons
    neurons.forEach(neuron => {
      const { x, y, activation, isActive, layer } = neuron;
      
      ctx.beginPath();
      ctx.arc(x, y, 20, 0, 2 * Math.PI);
      
      if (isActive) {
        // Create gradient for active neurons
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 25);
        gradient.addColorStop(0, `rgba(168, 85, 247, ${activation})`);
        gradient.addColorStop(1, 'rgba(168, 85, 247, 0.2)');
        ctx.fillStyle = gradient;
        
        // Add glow effect
        ctx.shadowColor = '#a855f7';
        ctx.shadowBlur = 20;
      } else {
        ctx.fillStyle = '#374151';
        ctx.shadowBlur = 0;
      }
      
      ctx.fill();
      
      // Border
      ctx.strokeStyle = isActive ? '#a855f7' : '#6b7280';
      ctx.lineWidth = 2;
      ctx.shadowBlur = 0;
      ctx.stroke();
      
      // Layer labels
      if (neuron.index === 0) {
        ctx.fillStyle = '#e2e8f0';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        
        let label = '';
        if (layer === 0) label = 'Input';
        else if (layer === neurons.filter(n => n.layer === layer).length) label = 'Output';
        else label = `Hidden ${layer}`;
        
        ctx.fillText(label, x, y - 40);
      }
    });
  }, [neurons, connections]);

  return (
    <Card className="bg-slate-800/50 border-purple-500/20 backdrop-blur-sm p-6">
      <div className="mb-4">
        <h3 className="text-xl font-bold text-purple-300 mb-2">Network Architecture</h3>
        <div className="flex items-center space-x-4 text-sm text-slate-300">
          <span>Input: {modelData.architecture.input_size}</span>
          <span>Hidden: [{modelData.architecture.hidden_layers.join(', ')}]</span>
          <span>Output: {modelData.architecture.output_size}</span>
          <span>Loss: {modelData.training_config.loss_function.toUpperCase()}</span>
        </div>
      </div>
      
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={800}
          height={400}
          className="w-full h-96 bg-slate-900 rounded-lg border border-slate-600"
        />
        
        {isInferenceRunning && (
          <div className="absolute top-4 right-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white px-3 py-1 rounded-full text-sm font-semibold animate-pulse">
            Processing...
          </div>
        )}
      </div>
      
      <div className="mt-4 text-center text-slate-400 text-sm">
        {isInferenceRunning ? 
          'Watch the data flow through your neural network!' : 
          'Make a prediction to see the network in action'
        }
      </div>
    </Card>
  );
};

export default NetworkVisualizer;
