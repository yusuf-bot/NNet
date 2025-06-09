
import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Upload, Brain, Settings } from 'lucide-react';

interface TrainingPanelProps {
  onModelTrained: (modelData: any) => void;
  onBack: () => void;
}

const TrainingPanel: React.FC<TrainingPanelProps> = ({ onModelTrained, onBack }) => {
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [hiddenLayers, setHiddenLayers] = useState('64 32');
  const [learningRate, setLearningRate] = useState('0.01');
  const [epochs, setEpochs] = useState('1000');
  const [lossFunction, setLossFunction] = useState('mse');
  const [isTraining, setIsTraining] = useState(false);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'text/csv') {
      setCsvFile(file);
    }
  };

  const handleTrain = async () => {
    if (!csvFile) return;

    setIsTraining(true);
    
    // Simulate training process
    setTimeout(() => {
      const hiddenLayersArray = hiddenLayers.split(' ').map(n => parseInt(n)).filter(n => !isNaN(n));
      
      const mockModelData = {
        architecture: {
          input_size: 3, // This would come from CSV analysis
          output_size: 1,
          hidden_layers: hiddenLayersArray.length > 0 ? hiddenLayersArray : [64, 32]
        },
        training_config: {
          activation_names: ['relu', ...hiddenLayersArray.map(() => 'relu'), 'identity'],
          loss_function: lossFunction,
          learning_rate: parseFloat(learningRate)
        }
      };
      
      setIsTraining(false);
      onModelTrained(mockModelData);
    }, 3000);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <Card className="bg-slate-800/50 border-purple-500/20 backdrop-blur-sm p-8">
        <div className="mb-8">
          <div className="flex items-center mb-4">
            <Brain className="w-8 h-8 text-purple-400 mr-3" />
            <h2 className="text-3xl font-bold text-purple-300">Train New Model</h2>
          </div>
          <p className="text-slate-300">Upload your training data and configure your neural network</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Data Upload */}
          <div className="space-y-6">
            <div>
              <Label className="text-lg font-semibold text-purple-300 mb-4 block">Training Data</Label>
              
              <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-purple-500 transition-colors">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="csv-upload"
                />
                <label htmlFor="csv-upload" className="cursor-pointer">
                  <Upload className="w-12 h-12 mx-auto mb-4 text-slate-400" />
                  <p className="text-slate-300 mb-2">
                    {csvFile ? csvFile.name : 'Click to upload CSV file'}
                  </p>
                  <p className="text-sm text-slate-500">
                    Upload your training dataset in CSV format
                  </p>
                </label>
              </div>
            </div>
          </div>

          {/* Configuration */}
          <div className="space-y-6">
            <div>
              <Label className="text-lg font-semibold text-purple-300 mb-4 block flex items-center">
                <Settings className="w-5 h-5 mr-2" />
                Network Configuration
              </Label>
              
              <div className="space-y-4">
                <div>
                  <Label htmlFor="hidden-layers" className="text-slate-300">Hidden Layers</Label>
                  <Input
                    id="hidden-layers"
                    value={hiddenLayers}
                    onChange={(e) => setHiddenLayers(e.target.value)}
                    placeholder="64 32"
                    className="bg-slate-700 border-slate-600 text-white"
                  />
                  <p className="text-xs text-slate-500 mt-1">Space-separated layer sizes (e.g., "64 32")</p>
                </div>

                <div>
                  <Label htmlFor="learning-rate" className="text-slate-300">Learning Rate</Label>
                  <Input
                    id="learning-rate"
                    value={learningRate}
                    onChange={(e) => setLearningRate(e.target.value)}
                    placeholder="0.01"
                    className="bg-slate-700 border-slate-600 text-white"
                  />
                </div>

                <div>
                  <Label htmlFor="epochs" className="text-slate-300">Epochs</Label>
                  <Input
                    id="epochs"
                    value={epochs}
                    onChange={(e) => setEpochs(e.target.value)}
                    placeholder="1000"
                    className="bg-slate-700 border-slate-600 text-white"
                  />
                </div>

                <div>
                  <Label htmlFor="loss-function" className="text-slate-300">Loss Function</Label>
                  <select
                    id="loss-function"
                    value={lossFunction}
                    onChange={(e) => setLossFunction(e.target.value)}
                    className="w-full p-2 bg-slate-700 border border-slate-600 rounded-md text-white"
                  >
                    <option value="mse">Mean Squared Error (MSE)</option>
                    <option value="mae">Mean Absolute Error (MAE)</option>
                    <option value="cross_entropy">Cross Entropy</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-between mt-8">
          <Button 
            onClick={onBack}
            variant="outline"
            className="border-slate-600 text-slate-300 hover:bg-slate-700"
          >
            ← Back to Menu
          </Button>
          
          <Button
            onClick={handleTrain}
            disabled={!csvFile || isTraining}
            className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold px-8"
          >
            {isTraining ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Training...
              </>
            ) : (
              'Start Training'
            )}
          </Button>
        </div>
      </Card>

      {isTraining && (
        <Card className="mt-6 bg-slate-800/50 border-green-500/20 backdrop-blur-sm p-6">
          <div className="text-center">
            <div className="animate-pulse">
              <Brain className="w-16 h-16 mx-auto mb-4 text-green-400" />
              <h3 className="text-xl font-bold text-green-300 mb-2">Training in Progress</h3>
              <p className="text-slate-300">Building your neural network from scratch...</p>
              
              <div className="mt-6 bg-slate-700 rounded-full h-2">
                <div className="bg-gradient-to-r from-green-400 to-blue-400 h-2 rounded-full animate-pulse" style={{width: '60%'}}></div>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default TrainingPanel;
