import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Upload, Calculator, FileText, Target } from 'lucide-react';
import { predict } from '../api';

interface ModelData {
  model_info: { created_at: string; version: string };
  architecture: { input_size: number; output_size: number; hidden_layers: number[] };
  training_config: { activation_names: string[]; loss_function: string; learning_rate: number };
  layers: any[];
  metadata: any;
}

interface PredictionPanelProps {
  modelData: ModelData;
  onInferenceStart: () => void;
  onInferenceEnd: () => void;
}

const PredictionPanel: React.FC<PredictionPanelProps> = ({
  modelData,
  onInferenceStart,
  onInferenceEnd,
}) => {
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single');
  const [singleInputs, setSingleInputs] = useState<string[]>(
    Array(modelData.architecture.input_size).fill('')
  );
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<number[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSingleInputChange = (index: number, value: string) => {
    const newInputs = [...singleInputs];
    newInputs[index] = value;
    setSingleInputs(newInputs);
  };

  const handleSinglePrediction = async () => {
    const inputs = singleInputs.map(val => parseFloat(val));
    if (inputs.some(isNaN)) {
      setError('All inputs must be valid numbers');
      return;
    }

    setIsLoading(true);
    setError(null);
    setPrediction(null);
    
    // Start the animation
    onInferenceStart();

    try {
      // Calculate minimum animation time based on number of layers
      const layers = [modelData.architecture.input_size, ...modelData.architecture.hidden_layers, modelData.architecture.output_size];
      const minAnimationTime = layers.length * 800 + 500; // 800ms per layer + 500ms buffer

      // Start both the API call and ensure minimum animation time
      const [result] = await Promise.all([
        predict(modelData, inputs),
        new Promise(resolve => setTimeout(resolve, minAnimationTime))
      ]);

      // Normalize prediction to flat array
      const normalizedPrediction = Array.isArray(result)
        ? result.flat().map(Number)
        : [Number(result)];
      
      setPrediction(normalizedPrediction);
    } catch (err: any) {
      setError(err.message || 'Prediction failed');
    } finally {
      setIsLoading(false);
      onInferenceEnd();
    }
  };

  const handleBatchPrediction = async () => {
    if (!batchFile) return;

    setIsLoading(true);
    setError(null);
    onInferenceStart();

    try {
      // Calculate minimum animation time
      const layers = [modelData.architecture.input_size, ...modelData.architecture.hidden_layers, modelData.architecture.output_size];
      const minAnimationTime = layers.length * 800 + 1000; // Extra time for batch processing

      // Simulate batch prediction (replace with actual API call)
      await new Promise(resolve => setTimeout(resolve, minAnimationTime));
      
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
    } catch (err: any) {
      setError(err.message || 'Batch prediction failed');
    } finally {
      setIsLoading(false);
      onInferenceEnd();
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'text/csv') {
      setBatchFile(file);
      setError(null);
    } else if (file) {
      setError('Please upload a CSV file');
    }
  };

  const clearInputs = () => {
    setSingleInputs(Array(modelData.architecture.input_size).fill(''));
    setPrediction(null);
    setError(null);
  };

  const fillRandomInputs = () => {
    const randomInputs = Array(modelData.architecture.input_size)
      .fill(0)
      .map(() => (Math.random() * 2 - 1).toFixed(3)); // Random values between -1 and 1
    setSingleInputs(randomInputs);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="p-6 bg-white">
      <div className="mb-6">
        <h3 className="text-xl font-semibold text-black mb-4">Make Predictions</h3>
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

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      )}

      {activeTab === 'single' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <Calculator className="w-5 h-5 text-black mr-2" />
              <Label className="text-black font-medium">Input Features</Label>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={fillRandomInputs}
                variant="outline"
                size="sm"
                disabled={isLoading}
                className="text-xs"
              >
                Random
              </Button>
              <Button
                onClick={clearInputs}
                variant="outline"
                size="sm"
                disabled={isLoading}
                className="text-xs"
              >
                Clear
              </Button>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {singleInputs.map((input, index) => (
              <div key={index}>
                <Label htmlFor={`input-${index}`} className="text-black text-sm">
                  Feature {index + 1}
                </Label>
                <Input
                  id={`input-${index}`}
                  type="number"
                  step="any"
                  value={input}
                  onChange={(e) => handleSingleInputChange(index, e.target.value)}
                  placeholder={`Value ${index + 1}`}
                  className="bg-white border-2 border-gray-300 text-black focus:border-black"
                  disabled={isLoading}
                />
              </div>
            ))}
          </div>
          
          <Button
            onClick={handleSinglePrediction}
            disabled={isLoading || singleInputs.some(val => val === '')}
            className="w-full bg-black hover:bg-gray-800 text-white font-medium mt-6"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Processing Neural Network...
              </>
            ) : (
              'Run Prediction'
            )}
          </Button>
          
          {prediction && (
            <div className="mt-6 p-4 bg-gray-100 rounded-lg border-2 border-gray-200">
              <div className="flex items-center mb-2">
                <Target className="w-5 h-5 text-green-600 mr-2" />
                <Label className="text-black font-medium">Prediction Result</Label>
              </div>
              <div className="text-2xl font-bold text-black">
                {prediction.length === 1 
                  ? prediction[0].toFixed(6)
                  : `[${prediction.map(val => val.toFixed(4)).join(', ')}]`
                }
              </div>
              {modelData.architecture.output_size > 1 && (
                <p className="text-sm text-gray-600 mt-2">
                  Output vector with {prediction.length} values
                </p>
              )}
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
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-black transition-colors">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
              id="batch-upload"
              disabled={isLoading}
            />
            <label htmlFor="batch-upload" className="cursor-pointer">
              <Upload className="w-8 h-8 mx-auto mb-2 text-gray-600" />
              <p className="text-black mb-1 font-medium">
                {batchFile ? batchFile.name : 'Click to upload CSV file'}
              </p>
              <p className="text-sm text-gray-600">
                Upload CSV with {modelData.architecture.input_size} input features per row
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
                Processing Batch...
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