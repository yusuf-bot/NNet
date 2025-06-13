import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Upload, Brain, Settings } from 'lucide-react';
import { trainModel } from '../api';

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
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'text/csv') {
      setCsvFile(file);
      setError(null);
    } else {
      setError('Please upload a valid CSV file');
    }
  };

  const validateInputs = () => {
    const layers = hiddenLayers.split(' ').map(n => parseInt(n)).filter(n => !isNaN(n));
    if (layers.length === 0) {
      return 'Hidden layers must contain at least one valid integer';
    }

    const lr = parseFloat(learningRate);
    if (isNaN(lr) || lr <= 0) {
      return 'Learning rate must be a positive number';
    }

    const ep = parseInt(epochs);
    if (isNaN(ep) || ep <= 0) {
      return 'Epochs must be a positive integer';
    }

    if (!['mse', 'mae', 'cross_entropy'].includes(lossFunction)) {
      return 'Invalid loss function selected';
    }

    return null;
  };

  const handleTrain = async () => {
    if (!csvFile) {
      setError('Please upload a CSV file');
      return;
    }

    const validationError = validateInputs();
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsTraining(true);
    setError(null);

    try {
      const response = await trainModel(
        csvFile,
        hiddenLayers,
        learningRate,
        epochs,
        lossFunction
      );
      setIsTraining(false);
      onModelTrained(response);
    } catch (err: any) {
      console.error('Training error:', err);
      setIsTraining(false);
      setError(err.message || 'Training failed');
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <Card className="border-2 p-8">
        <div className="mb-8">
          <div className="flex items-center mb-4">
            <Brain className="w-8 h-8 text-foreground mr-3" />
            <h2 className="text-3xl font-bold text-foreground">Train New Model</h2>
          </div>
         <div className="flex items-start mb-6">
  <p className="text-base text-gray-700 font-medium leading-relaxed">
    <span className="font-semibold text-black">NOTE:</span> Ensure that the CSV is properly configured with input fields labeled <span className="font-semibold text-black">"x1"</span> and <span className="font-semibold text-black">"x2"</span>, and output fields, if present, labeled <span className="font-semibold text-black">"y1"</span> and <span className="font-semibold text-black">"y2"</span>. The model performs optimally with supervised learning but can also support unsupervised learning.
  </p>
</div>

        
          <p className="text-muted-foreground">Upload your training data and configure your neural network</p>
          
        
      </div>

        {error && (
          <div className="mb-4 text-red-500">
            {error}
          </div>
        )}

        <div className="grid md:grid-cols-2 gap-8">
          {/* Data Upload */}
          <div className="space-y-6">
            <div>
              <Label className="text-lg font-semibold text-foreground mb-4 block">Training Data</Label>
              
              <div className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-foreground transition-colors">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="csv-upload"
                />
                <label htmlFor="csv-upload" className="cursor-pointer">
                  <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-foreground mb-2 font-medium">
                    {csvFile ? csvFile.name : 'Click to upload CSV file'}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Upload your training dataset in CSV format
                  </p>
                </label>
              </div>
            </div>
          </div>

          {/* Configuration */}
          <div className="space-y-6">
            <div>
              <Label className="text-lg font-semibold text-foreground mb-4 block flex items-center">
                <Settings className="w-5 h-5 mr-2" />
                Network Configuration
              </Label>
              
              <div className="space-y-4">
                <div>
                  <Label htmlFor="hidden-layers" className="text-foreground font-medium">Hidden Layers</Label>
                  <Input
                    id="hidden-layers"
                    value={hiddenLayers}
                    onChange={(e) => setHiddenLayers(e.target.value)}
                    placeholder="64 32"
                    className="bg-background border-2 border-border text-foreground focus:border-foreground"
                  />
                  <p className="text-xs text-muted-foreground mt-1">Space-separated layer sizes (e.g., "64 32")</p>
                </div>

                <div>
                  <Label htmlFor="learning-rate" className="text-foreground font-medium">Learning Rate</Label>
                  <Input
                    id="learning-rate"
                    value={learningRate}
                    onChange={(e) => setLearningRate(e.target.value)}
                    placeholder="0.01"
                    className="bg-background border-2 border-border text-foreground focus:border-foreground"
                  />
                </div>

                <div>
                  <Label htmlFor="epochs" className="text-foreground font-medium">Epochs</Label>
                  <Input
                    id="epochs"
                    value={epochs}
                    onChange={(e) => setEpochs(e.target.value)}
                    placeholder="1000"
                    className="bg-background border-2 border-border text-foreground focus:border-foreground"
                  />
                </div>

                <div>
                  <Label htmlFor="loss-function" className="text-foreground font-medium">Loss Function</Label>
                  <select
                    id="loss-function"
                    value={lossFunction}
                    onChange={(e) => setLossFunction(e.target.value)}
                    className="w-full p-2 bg-background border-2 border-border rounded-md text-foreground focus:border-foreground focus:outline-none"
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
            className="border-2 border-foreground text-foreground hover:bg-foreground hover:text-background"
          >
            ← Back to Menu
          </Button>
          
          <Button
            onClick={handleTrain}
            disabled={!csvFile || isTraining}
            className="bg-foreground hover:bg-foreground/90 text-background font-medium px-8"
          >
            {isTraining ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-background mr-2"></div>
                Training...
              </>
            ) : (
              'Start Training'
            )}
          </Button>
        </div>
      </Card>

      {isTraining && (
        <Card className="mt-6 border-2 p-6">
          <div className="text-center">
            <div className="animate-pulse">
              <Brain className="w-16 h-16 mx-auto mb-4 text-foreground" />
              <h3 className="text-xl font-bold text-foreground mb-2">Training in Progress</h3>
              <p className="text-muted-foreground">Building your neural network from scratch...</p>
              
              <div className="mt-6 bg-muted rounded-full h-2">
                <div className="bg-foreground h-2 rounded-full animate-pulse" style={{width: '60%'}}></div>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default TrainingPanel;