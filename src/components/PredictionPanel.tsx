
import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Play, Upload, Download } from 'lucide-react';

interface PredictionPanelProps {
  modelData: any;
  onInferenceStart: () => void;
  onInferenceEnd: () => void;
}

const PredictionPanel: React.FC<PredictionPanelProps> = ({ 
  modelData, 
  onInferenceStart, 
  onInferenceEnd 
}) => {
  const [singleInputs, setSingleInputs] = useState<string[]>(
    Array(modelData.architecture.input_size).fill('')
  );
  const [predictionResult, setPredictionResult] = useState<number | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [batchResults, setBatchResults] = useState<number[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleSinglePrediction = async () => {
    const inputs = singleInputs.map(input => parseFloat(input)).filter(n => !isNaN(n));
    
    if (inputs.length !== modelData.architecture.input_size) {
      alert(`Please provide exactly ${modelData.architecture.input_size} input values`);
      return;
    }

    setIsProcessing(true);
    onInferenceStart();

    // Simulate prediction
    setTimeout(() => {
      const mockPrediction = Math.random() * 100; // Mock prediction result
      setPredictionResult(mockPrediction);
      setIsProcessing(false);
      onInferenceEnd();
    }, 2000);
  };

  const handleBatchPrediction = async () => {
    if (!csvFile) return;

    setIsProcessing(true);
    onInferenceStart();

    // Simulate batch prediction
    setTimeout(() => {
      const mockResults = Array.from({ length: 10 }, () => Math.random() * 100);
      setBatchResults(mockResults);
      setIsProcessing(false);
      onInferenceEnd();
    }, 3000);
  };

  const handleInputChange = (index: number, value: string) => {
    const newInputs = [...singleInputs];
    newInputs[index] = value;
    setSingleInputs(newInputs);
  };

  const downloadResults = () => {
    const csvContent = batchResults.map((result, index) => `${index + 1},${result.toFixed(4)}`).join('\n');
    const blob = new Blob([`Row,Prediction\n${csvContent}`], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'predictions.csv';
    a.click();
  };

  return (
    <Card className="bg-slate-800/50 border-purple-500/20 backdrop-blur-sm p-6 h-fit">
      <h3 className="text-xl font-bold text-purple-300 mb-4">Make Predictions</h3>
      
      <Tabs defaultValue="single" className="w-full">
        <TabsList className="grid w-full grid-cols-2 bg-slate-700">
          <TabsTrigger value="single" className="data-[state=active]:bg-purple-600">Single</TabsTrigger>
          <TabsTrigger value="batch" className="data-[state=active]:bg-purple-600">Batch</TabsTrigger>
        </TabsList>
        
        <TabsContent value="single" className="space-y-4 mt-6">
          <div className="space-y-3">
            <Label className="text-slate-300">Input Features</Label>
            {singleInputs.map((input, index) => (
              <div key={index}>
                <Input
                  type="number"
                  placeholder={`Feature ${index + 1}`}
                  value={input}
                  onChange={(e) => handleInputChange(index, e.target.value)}
                  className="bg-slate-700 border-slate-600 text-white"
                />
              </div>
            ))}
          </div>
          
          <Button
            onClick={handleSinglePrediction}
            disabled={isProcessing}
            className="w-full bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white"
          >
            {isProcessing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Processing...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Predict
              </>
            )}
          </Button>
          
          {predictionResult !== null && (
            <Card className="bg-gradient-to-r from-green-500/20 to-emerald-500/20 border-green-500/30 p-4 mt-4">
              <div className="text-center">
                <Label className="text-green-300 text-sm">Prediction Result</Label>
                <div className="text-2xl font-bold text-green-200 mt-1">
                  {predictionResult.toFixed(4)}
                </div>
              </div>
            </Card>
          )}
        </TabsContent>
        
        <TabsContent value="batch" className="space-y-4 mt-6">
          <div>
            <Label className="text-slate-300 mb-2 block">Upload CSV File</Label>
            <div className="border-2 border-dashed border-slate-600 rounded-lg p-4 text-center hover:border-purple-500 transition-colors">
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setCsvFile(e.target.files?.[0] || null)}
                className="hidden"
                id="batch-csv"
              />
              <label htmlFor="batch-csv" className="cursor-pointer">
                <Upload className="w-8 h-8 mx-auto mb-2 text-slate-400" />
                <p className="text-sm text-slate-300">
                  {csvFile ? csvFile.name : 'Click to upload CSV'}
                </p>
              </label>
            </div>
          </div>
          
          <Button
            onClick={handleBatchPrediction}
            disabled={!csvFile || isProcessing}
            className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white"
          >
            {isProcessing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Processing...
              </>
            ) : (
              'Process Batch'
            )}
          </Button>
          
          {batchResults.length > 0 && (
            <Card className="bg-slate-700/50 border-slate-600 p-4 mt-4">
              <div className="flex justify-between items-center mb-3">
                <Label className="text-slate-300">Batch Results ({batchResults.length} predictions)</Label>
                <Button
                  onClick={downloadResults}
                  size="sm"
                  variant="outline"
                  className="border-slate-600 text-slate-300 hover:bg-slate-600"
                >
                  <Download className="w-4 h-4 mr-1" />
                  Download
                </Button>
              </div>
              
              <div className="max-h-40 overflow-y-auto space-y-1">
                {batchResults.slice(0, 5).map((result, index) => (
                  <div key={index} className="flex justify-between text-sm">
                    <span className="text-slate-400">Row {index + 1}:</span>
                    <span className="text-white font-mono">{result.toFixed(4)}</span>
                  </div>
                ))}
                {batchResults.length > 5 && (
                  <div className="text-center text-slate-500 text-xs">
                    ... and {batchResults.length - 5} more results
                  </div>
                )}
              </div>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </Card>
  );
};

export default PredictionPanel;
