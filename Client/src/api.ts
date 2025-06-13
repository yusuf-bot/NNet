import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000',
});

interface PredictResponse {
  prediction: number[];
  error?: string;
}

export const trainModel = async (
  csvFile: File,
  hiddenLayers: string,
  learningRate: string,
  epochs: string,
  lossFunction: string
) => {
  const formData = new FormData();
  formData.append('file', csvFile);
  formData.append('hidden_layers', hiddenLayers);
  formData.append('learning_rate', learningRate);
  formData.append('epochs', epochs);
  formData.append('loss_function', lossFunction);

  try {
    const response = await api.post('/train', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error: any) {
    throw new Error(error.response?.data?.error || 'Failed to train model');
  }
};

export const predict = async (model: any, inputs: number[]): Promise<number[]> => {
  try {
    const response = await api.post<PredictResponse>('/predict', { model, inputs }, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    return response.data.prediction;
  } catch (error: any) {
    throw new Error(error.response?.data?.error || 'Failed to make prediction');
  }
};