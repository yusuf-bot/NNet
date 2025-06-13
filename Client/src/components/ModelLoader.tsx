
import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Upload, FileText, CheckCircle } from 'lucide-react';

interface ModelLoaderProps {
  onModelLoaded: (modelData: any) => void;
  onBack: () => void;
}

const ModelLoader: React.FC<ModelLoaderProps> = ({ onModelLoaded, onBack }) => {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [modelPreview, setModelPreview] = useState<any>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.name.endsWith('.nn')) {
      setModelFile(file);
      
      try {
        const text = await file.text();
        const modelData = JSON.parse(text);
        setModelPreview(modelData);
      } catch (error) {
        console.error('Error parsing model file:', error);
      }
    }
  };

  const handleLoadModel = () => {
    if (!modelPreview) return;

    setIsLoading(true);
    
    setTimeout(() => {
      setIsLoading(false);
      onModelLoaded(modelPreview);
    }, 1500);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <Card className="border-2 p-8">
        <div className="mb-8">
          <div className="flex items-center mb-4">
            <Upload className="w-8 h-8 text-foreground mr-3" />
            <h2 className="text-3xl font-bold text-foreground">Load Existing Model</h2>
          </div>
          <p className="text-muted-foreground">Upload a pre-trained .nn model file to start making predictions</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* File Upload */}
          <div>
            <div className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-foreground transition-colors">
              <input
                type="file"
                accept=".nn"
                onChange={handleFileUpload}
                className="hidden"
                id="model-upload"
              />
              <label htmlFor="model-upload" className="cursor-pointer">
                <FileText className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                <p className="text-foreground mb-2 font-medium">
                  {modelFile ? modelFile.name : 'Click to upload .nn file'}
                </p>
                <p className="text-sm text-muted-foreground">
                  Select your trained neural network model
                </p>
              </label>
            </div>

            {modelFile && !isLoading && (
              <div className="mt-6 flex items-center justify-center text-foreground">
                <CheckCircle className="w-5 h-5 mr-2" />
                <span className="font-medium">Model file loaded successfully</span>
              </div>
            )}
          </div>

          {/* Model Preview */}
          <div>
            {modelPreview && (
              <Card className="border-2 p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">Model Preview</h3>
                
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Input Size:</span>
                    <span className="text-foreground font-medium">{modelPreview.architecture?.input_size}</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Hidden Layers:</span>
                    <span className="text-foreground font-medium">[{modelPreview.architecture?.hidden_layers?.join(', ')}]</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Output Size:</span>
                    <span className="text-foreground font-medium">{modelPreview.architecture?.output_size}</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Loss Function:</span>
                    <span className="text-foreground font-medium">{modelPreview.training_config?.loss_function?.toUpperCase()}</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Learning Rate:</span>
                    <span className="text-foreground font-medium">{modelPreview.training_config?.learning_rate}</span>
                  </div>
                </div>
              </Card>
            )}
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
            onClick={handleLoadModel}
            disabled={!modelPreview || isLoading}
            className="bg-foreground hover:bg-foreground/90 text-background font-medium px-8"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-background mr-2"></div>
                Loading...
              </>
            ) : (
              'Load Model'
            )}
          </Button>
        </div>
      </Card>

      {isLoading && (
        <Card className="mt-6 border-2 p-6">
          <div className="text-center">
            <div className="animate-pulse">
              <Upload className="w-16 h-16 mx-auto mb-4 text-foreground" />
              <h3 className="text-xl font-bold text-foreground mb-2">Loading Model</h3>
              <p className="text-muted-foreground">Initializing your neural network...</p>
              
              <div className="mt-6 bg-muted rounded-full h-2">
                <div className="bg-foreground h-2 rounded-full animate-pulse" style={{width: '80%'}}></div>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default ModelLoader;
