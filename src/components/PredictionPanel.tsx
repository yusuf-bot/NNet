
import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Upload, Calculator, FileText, Target } from 'lucide-react';

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

interface PredictionPanelProps {
  modelData: ModelData;
  onInferenceStart: () => void;
  onInferenceEnd: () => void;
}

const PredictionPanel: React.FC<PredictionPanelProps> = ({ 
  modelData, 
  onInferenceStart, 
  onInferenceEnd 
}) => {
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single');
  const [singleInputs, setSingleInputs] = useState<string[]>(
    Array(modelData.architecture.input_size).fill('')
  );
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<number[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSingleInputChange = (index: number, value: string) => {
    const newInputs = [...singleInputs];
    newInputs[index] = value;
    setSingleInputs(newInputs);
  };

  const handleSinglePrediction = async () => {
    const inputs = singleInputs.map(val => parseFloat(val) || 0);
    
    setIsLoading(true);
    onInferenceStart();
    
    // Simulate prediction
    setTimeout(() => {
      const mockPrediction = inputs.map(val => val * Math.random() + 0.5);
      setPrediction(mockPrediction);
      setIsLoading(false);
      onInferenceEnd();
    }, 2000);
  };

  const handleBatchPrediction = async () => {
    if (!batchFile) return;
    
    setIsLoading(true);
    onInferenceStart();
    
    // Simulate batch prediction
    setTimeout(() => {
      // Create download for results
      const results = 'input1,input2,input3,prediction\n1.0,2.0,3.0,0.85\n2.0,3.0,4.0,0.92\n';
      const blob = new Blob([results], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'predictions.csv';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      setIsLoading(false);
      onInferenceEnd();
    }, 2000);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'text/csv') {
      setBatchFile(file);
    }
  };

  return (
    <div className="border-2 border-border rounded-lg p-6 bg-background">
      <div className="mb-6">
        <h3 className="text-xl font-semibold text-foreground mb-4">Make Predictions</h3>
        
        {/* Tabs */}
        <div className="flex border-b border-border">
          <button
            onClick={() => setActiveTab('single')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'single'
                ? 'border-foreground text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            Single Prediction
          </button>
          <button
            onClick={() => setActiveTab('batch')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'batch'
                ? 'border-foreground text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            Batch Prediction
          </button>
        </div>
      </div>

      {activeTab === 'single' && (
        <div className="space-y-4">
          <div className="flex items-center mb-4">
            <Calculator className="w-5 h-5 text-foreground mr-2" />
            <Label className="text-foreground font-medium">Input Features</Label>
          </div>
          
          {singleInputs.map((input, index) => (
            <div key={index}>
              <Label htmlFor={`input-${index}`} className="text-foreground">
                Feature {index + 1}
              </Label>
              <Input
                id={`input-${index}`}
                value={input}
                onChange={(e) => handleSingleInputChange(index, e.target.value)}
                placeholder={`Enter value for feature ${index + 1}`}
                className="bg-background border-2 border-border text-foreground focus:border-foreground"
              />
            </div>
          ))}
          
          <Button
            onClick={handleSinglePrediction}
            disabled={isLoading || singleInputs.some(val => val === '')}
            className="w-full bg-foreground hover:bg-foreground/90 text-background font-medium mt-4"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-background mr-2"></div>
                Predicting...
              </>
            ) : (
              'Predict'
            )}
          </Button>
          
          {prediction && (
            <div className="mt-6 p-4 bg-muted rounded-lg">
              <div className="flex items-center mb-2">
                <Target className="w-5 h-5 text-foreground mr-2" />
                <Label className="text-foreground font-medium">Prediction Result</Label>
              </div>
              <div className="text-2xl font-bold text-foreground">
                {prediction[0]?.toFixed(4)}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'batch' && (
        <div className="space-y-4">
          <div className="flex items-center mb-4">
            <FileText className="w-5 h-5 text-foreground mr-2" />
            <Label className="text-foreground font-medium">Batch Prediction</Label>
          </div>
          
          <div className="border-2 border-dashed border-border rounded-lg p-6 text-center hover:border-foreground transition-colors">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
              id="batch-upload"
            />
            <label htmlFor="batch-upload" className="cursor-pointer">
              <Upload className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
              <p className="text-foreground mb-1 font-medium">
                {batchFile ? batchFile.name : 'Click to upload CSV file'}
              </p>
              <p className="text-sm text-muted-foreground">
                Upload CSV with input features (no headers)
              </p>
            </label>
          </div>
          
          <Button
            onClick={handleBatchPrediction}
            disabled={!batchFile || isLoading}
            className="w-full bg-foreground hover:bg-foreground/90 text-background font-medium"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-background mr-2"></div>
                Processing...
              </>
            ) : (
              'Run Batch Prediction'
            )}
          </Button>
          
          <p className="text-xs text-muted-foreground text-center">
            Results will be automatically downloaded as CSV
          </p>
        </div>
      )}
    </div>
  );
};

export default PredictionPanel;
