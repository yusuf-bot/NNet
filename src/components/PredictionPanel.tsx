import React, { useState } from 'react';
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
    <div className="p-6 bg-white">
      <div className="mb-6">
        <h3 className="text-xl font-semibold text-black mb-4">Make Predictions</h3>
        
        {/* Tabs */}
        <div className="flex border-b border-gray-300">
          <button
            onClick={() => setActiveTab('single')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'single'
                ? 'border-black text-black'
                : 'border-transparent text-gray-600 hover:text-black'
            }`}
          >
            Single Prediction
          </button>
          <button
            onClick={() => setActiveTab('batch')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'batch'
                ? 'border-black text-black'
                : 'border-transparent text-gray-600 hover:text-black'
            }`}
          >
            Batch Prediction
          </button>
        </div>
      </div>

      {activeTab === 'single' && (
        <div className="space-y-4">
          <div className="flex items-center mb-4">
            <Calculator className="w-5 h-5 text-black mr-2" />
            <Label className="text-black font-medium">Input Features</Label>
          </div>
          
          {singleInputs.map((input, index) => (
            <div key={index}>
              <Label htmlFor={`input-${index}`} className="text-black">
                Feature {index + 1}
              </Label>
              <Input
                id={`input-${index}`}
                value={input}
                onChange={(e) => handleSingleInputChange(index, e.target.value)}
                placeholder={`Enter value for feature ${index + 1}`}
                className="bg-white border-2 border-gray-300 text-black focus:border-black"
              />
            </div>
          ))}
          
          <Button
            onClick={handleSinglePrediction}
            disabled={isLoading || singleInputs.some(val => val === '')}
            className="w-full bg-black hover:bg-gray-800 text-white font-medium mt-4"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Predicting...
              </>
            ) : (
              'Predict'
            )}
          </Button>
          
          {prediction && (
            <div className="mt-6 p-4 bg-gray-100 rounded-lg">
              <div className="flex items-center mb-2">
                <Target className="w-5 h-5 text-black mr-2" />
                <Label className="text-black font-medium">Prediction Result</Label>
              </div>
              <div className="text-2xl font-bold text-black">
                {prediction[0]?.toFixed(4)}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'batch' && (
        <div className="space-y-4">
          <div className="flex items-center mb-4">
            <FileText className="w-5 h-5 text-black mr-2" />
            <Label className="text-black font-medium">Batch Prediction</Label>
          </div>
          
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-black transition-colors">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
              id="batch-upload"
            />
            <label htmlFor="batch-upload" className="cursor-pointer">
              <Upload className="w-8 h-8 mx-auto mb-2 text-gray-600" />
              <p className="text-black mb-1 font-medium">
                {batchFile ? batchFile.name : 'Click to upload CSV file'}
              </p>
              <p className="text-sm text-gray-600">
                Upload CSV with input features (no headers)
              </p>
            </label>
          </div>
          
          <Button
            onClick={handleBatchPrediction}
            disabled={!batchFile || isLoading}
            className="w-full bg-black hover:bg-gray-800 text-white font-medium"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Processing...
              </>
            ) : (
              'Run Batch Prediction'
            )}
          </Button>
          
          <p className="text-xs text-gray-600 text-center">
            Results will be automatically downloaded as CSV
          </p>
        </div>
      )}
    </div>
  );
};

export default PredictionPanel;
