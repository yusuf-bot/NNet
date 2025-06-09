
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
  isVisible: boolean;
}

interface Connection {
  from: Neuron;
  to: Neuron;
  weight: number;
  isActive: boolean;
}

const MAX_VISIBLE_NEURONS = 8;

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
    
    // Create neurons with smart visibility
    layers.forEach((layerSize, layerIndex) => {
      const visibleNeurons = Math.min(layerSize, MAX_VISIBLE_NEURONS);
      const showEllipsis = layerSize > MAX_VISIBLE_NEURONS;
      const actualVisibleCount = showEllipsis ? visibleNeurons - 1 : visibleNeurons;
      
      const neuronSpacing = height / (visibleNeurons + 1);
      
      for (let i = 0; i < layerSize; i++) {
        const isVisible = i < actualVisibleCount || (showEllipsis && i === layerSize - 1);
        const displayIndex = i < actualVisibleCount ? i : actualVisibleCount;
        
        newNeurons.push({
          x: layerSpacing * (layerIndex + 1),
          y: neuronSpacing * (displayIndex + 1),
          layer: layerIndex,
          index: i,
          activation: 0,
          isActive: false,
          isVisible
        });
      }
    });
    
    // Create connections only between visible neurons
    for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex++) {
      const currentLayerNeurons = newNeurons.filter(n => n.layer === layerIndex && n.isVisible);
      const nextLayerNeurons = newNeurons.filter(n => n.layer === layerIndex + 1 && n.isVisible);
      
      currentLayerNeurons.forEach(fromNeuron => {
        nextLayerNeurons.forEach(toNeuron => {
          newConnections.push({
            from: fromNeuron,
            to: toNeuron,
            weight: Math.random() * 2 - 1,
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
      setNeurons(prev => prev.map(n => ({ ...n, activation: 0, isActive: false })));
      setConnections(prev => prev.map(c => ({ ...c, isActive: false })));
      return;
    }

    const animationInterval = setInterval(() => {
      setAnimationStep(prev => {
        const maxLayers = modelData.architecture.hidden_layers.length + 2;
        if (prev >= maxLayers) {
          return maxLayers;
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
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw connections
    connections.forEach(connection => {
      const { from, to, weight, isActive } = connection;
      
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      
      if (isActive) {
        ctx.strokeStyle = weight > 0 ? '#000000' : '#666666';
        ctx.lineWidth = Math.abs(weight) * 2 + 1;
        ctx.globalAlpha = 0.8;
      } else {
        ctx.strokeStyle = '#e5e5e5';
        ctx.lineWidth = 1;
        ctx.globalAlpha = 0.3;
      }
      
      ctx.stroke();
      ctx.globalAlpha = 1;
    });

    // Draw neurons
    const { input_size, output_size, hidden_layers } = modelData.architecture;
    const layers = [input_size, ...hidden_layers, output_size];
    
    neurons.filter(n => n.isVisible).forEach(neuron => {
      const { x, y, activation, isActive, layer } = neuron;
      
      ctx.beginPath();
      ctx.arc(x, y, 16, 0, 2 * Math.PI);
      
      if (isActive) {
        ctx.fillStyle = `rgba(0, 0, 0, ${0.2 + activation * 0.6})`;
        ctx.shadowColor = '#000000';
        ctx.shadowBlur = 8;
      } else {
        ctx.fillStyle = '#f5f5f5';
        ctx.shadowBlur = 0;
      }
      
      ctx.fill();
      
      // Border
      ctx.strokeStyle = isActive ? '#000000' : '#cccccc';
      ctx.lineWidth = 2;
      ctx.shadowBlur = 0;
      ctx.stroke();
    });

    // Draw ellipsis and layer info
    layers.forEach((layerSize, layerIndex) => {
      if (layerSize > MAX_VISIBLE_NEURONS) {
        const layerSpacing = canvas.width / (layers.length + 1);
        const x = layerSpacing * (layerIndex + 1);
        const neuronSpacing = canvas.height / (Math.min(layerSize, MAX_VISIBLE_NEURONS) + 1);
        const ellipsisY = neuronSpacing * (MAX_VISIBLE_NEURONS - 1);
        
        ctx.fillStyle = '#666666';
        ctx.font = '14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`(${layerSize - MAX_VISIBLE_NEURONS + 1} more...)`, x, ellipsisY);
      }
      
      // Layer labels
      const layerSpacing = canvas.width / (layers.length + 1);
      const x = layerSpacing * (layerIndex + 1);
      
      ctx.fillStyle = '#000000';
      ctx.font = 'bold 16px Inter, sans-serif';
      ctx.textAlign = 'center';
      
      let label = '';
      if (layerIndex === 0) label = 'Input';
      else if (layerIndex === layers.length - 1) label = 'Output';
      else label = `Hidden ${layerIndex}`;
      
      ctx.fillText(label, x, 30);
      ctx.font = '12px Inter, sans-serif';
      ctx.fillStyle = '#666666';
      ctx.fillText(`(${layerSize})`, x, 50);
    });
  }, [neurons, connections, modelData]);

  return (
    <Card className="border-2 p-6">
      <div className="mb-4">
        <h3 className="text-xl font-semibold text-foreground mb-2">Network Architecture</h3>
        <div className="flex items-center space-x-4 text-sm text-muted-foreground">
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
          className="w-full h-96 bg-white rounded-lg border-2 border-border"
        />
        
        {isInferenceRunning && (
          <div className="absolute top-4 right-4 bg-foreground text-background px-3 py-1 rounded-full text-sm font-medium animate-pulse">
            Processing...
          </div>
        )}
      </div>
      
      <div className="mt-4 text-center text-muted-foreground text-sm">
        {isInferenceRunning ? 
          'Watch the data flow through your neural network!' : 
          'Make a prediction to see the network in action'
        }
      </div>
    </Card>
  );
};

export default NetworkVisualizer;
