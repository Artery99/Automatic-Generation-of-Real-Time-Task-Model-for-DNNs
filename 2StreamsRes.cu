__global__ void convolutionalLayer(float* input, float* output, float kernel, int inputChannels, int inputHeight, int inputWidth, int outputChannels, int outputHeight, int outputWidth, int kernelSize, int stride) {
    // Compute thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute input and output indices
    int inputIndex = tid / (outputHeight * outputWidth);
    int outputIndex = tid % (outputHeight * outputWidth);

    // Compute input and output coordinates
    int outputX = outputIndex % outputWidth;
    int outputY = (outputIndex / outputWidth) % outputHeight;
    int inputX = outputX * stride;
    int inputY = outputY * stride;

    // Convolution operation
    for (int channel = 0; channel < outputChannels; ++channel) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernelSize; ++ky) {
            for (int kx = 0; kx < kernelSize; ++kx) {
                int inputPixelX = inputX + kx;
                int inputPixelY = inputY + ky;
                int inputPixelIndex = (inputIndex * inputHeight + inputPixelY) * inputWidth + inputPixelX;
                int kernelIndex = ((channel * kernelSize + ky) * kernelSize + kx) * inputChannels + inputIndex;
                sum += input[inputPixelIndex] * kernel[kernelIndex];
            }
        }
        output[tid * outputChannels + channel] = sum;
    }
}

__global__ void averagePoolingLayer(float* input, float* output, int inputChannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth) {
    // Compute thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute input and output indices
    int inputIndex = tid / (outputHeight * outputWidth);
    int outputIndex = tid % (outputHeight * outputWidth);

    // Compute input and output coordinates
    int outputX = outputIndex % outputWidth;
    int outputY = (outputIndex / outputWidth) % outputHeight;
    int inputX = outputX * 2;
    int inputY = outputY * 2;

    // Average pooling operation
    float avg = 0.0f;
    for (int ky = 0; ky < 2; ++ky) {
        for (int kx = 0; kx < 2; ++kx) {
            int inputPixelX = inputX + kx;
            int inputPixelY = inputY + ky;
            int inputPixelIndex = (inputIndex * inputHeight + inputPixelY) * inputWidth + inputPixelX;
            avg += input[inputPixelIndex];
        }
    }
    output[tid] = avg / 4.0f;
}

__global__ void fullyConnectedLayer(float* input, float* output, int inputSize, int outputSize) {
    // Compute thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute input and output indices
    int inputIndex = tid / outputSize;
    int outputIndex = tid % outputSize;

    // Fully connected operation
    float sum = 0.0f;
    for (int i = 0; i < inputSize; ++i) {
        sum += input[inputIndex * inputSize + i];
    }
    output[tid] = sum;
}

__global__ void softmaxActivation(float* input, int inputSize) {
    // Compute thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute input index
    int inputIndex = tid / inputSize;

    // Compute softmax activation
    float maxVal = 0.0f;
    for (int i = 0; i < inputSize; ++i) {
        int index = inputIndex * inputSize + i;
        maxVal = (input[index] > maxVal) ? input[index] : maxVal;
    }

    float sumExp = 0.0f;
    for (int i = 0; i < inputSize; ++i) {
        int index = inputIndex * inputSize + i;
        float val = expf(input[index] - maxVal);
        input[index] = val;
        sumExp += val;
    }

    for (int i = 0; i < inputSize; ++i) {
        int index = inputIndex * inputSize + i;
        input[index] /= sumExp;
    }
}

void resnet18(float* input, float* output, int inputChannels, int inputHeight, int inputWidth, int outputClasses, cudaStream_t stream) {
    // Allocate memory for intermediate feature maps
    float* intermediate1;
    cudaMalloc((void**)&intermediate1, inputChannels * inputHeight * inputWidth * sizeof(float));
    float* intermediate2;
    cudaMalloc((void**)&intermediate2, inputChannels * inputHeight * inputWidth * sizeof(float));

    // Launch convolutional layers
    int numThreads = inputChannels * inputHeight * inputWidth;
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(input, intermediate1, inputChannels, inputHeight, inputWidth, 64, inputHeight, inputWidth, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate1, intermediate2, 64, inputHeight, inputWidth, 64, inputHeight, inputWidth, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate2, intermediate1, 64, inputHeight, inputWidth, 64, inputHeight, inputWidth, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate1, intermediate2, 64, inputHeight, inputWidth, 64, inputHeight, inputWidth, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate2, intermediate1, 64, inputHeight, inputWidth, 64, inputHeight, inputWidth, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate1, intermediate2, 64, inputHeight, inputWidth, 128, inputHeight / 2, inputWidth / 2, 3, 2);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate2, intermediate1, 128, inputHeight / 2, inputWidth / 2, 128, inputHeight / 2, inputWidth / 2, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0,stream>>>(intermediate1, intermediate2, 128, inputHeight / 2, inputWidth / 2, 128, inputHeight / 2, inputWidth / 2, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate2, intermediate1, 128, inputHeight / 2, inputWidth / 2, 128, inputHeight / 2, inputWidth / 2, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate1, intermediate2, 128, inputHeight / 2, inputWidth / 2, 256, inputHeight / 4, inputWidth / 4, 3, 2);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate2, intermediate1, 256, inputHeight / 4, inputWidth / 4, 256, inputHeight / 4, inputWidth / 4, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate1, intermediate2, 256, inputHeight / 4, inputWidth / 4, 256, inputHeight / 4, inputWidth / 4, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate2, intermediate1, 256, inputHeight / 4, inputWidth / 4, 256, inputHeight / 4, inputWidth / 4, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate1, intermediate2, 256, inputHeight / 4, inputWidth / 4, 512, inputHeight / 8, inputWidth / 8, 3, 2);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate2, intermediate1, 512, inputHeight / 8, inputWidth / 8, 512, inputHeight / 8, inputWidth / 8, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate1, intermediate2, 512, inputHeight / 8, inputWidth / 8, 512, inputHeight / 8, inputWidth / 8, 3, 1);
    convolutionalLayer<<<(numThreads + 255) / 256, 256, 0, stream>>>(intermediate2, intermediate1, 512, inputHeight / 8, inputWidth / 8, 512, inputHeight / 8, inputWidth / 8, 3, 1);

    // Launch average pooling layer
    int poolingOutputHeight = inputHeight / 8;
    int poolingOutputWidth = inputWidth / 8;
    int poolingOutputSize = 512 * poolingOutputHeight * poolingOutputWidth;
    averagePoolingLayer<<<(poolingOutputSize + 255) / 256, 256, 0, stream>>>(intermediate1, output, 512, inputHeight / 8, inputWidth / 8, poolingOutputHeight, poolingOutputWidth);

    // Launch fully connected layer
    int fcInputSize = 512 * poolingOutputHeight * poolingOutputWidth;
    int fcOutputSize = outputClasses;
    int fcOutputSizePerThread = (fcOutputSize + 255) / 256;
    fullyConnectedLayer<<<(fcInputSize * fcOutputSizePerThread + 255) / 256, 256, 0, stream>>>(output, intermediate2, fcInputSize, fcOutputSize);

    // Launch softmax activation
    softmaxActivation<<<(fcInputSize * fcOutputSizePerThread + 255) / 256, 256, 0, stream>>>(intermediate2, fcOutputSize);

    // Free allocated memory
    cudaFree(intermediate1);
    cudaFree(intermediate2);
}

int main() {
    // Input dimensions
    int inputChannels = 3;
    int inputHeight = 224;
    int inputWidth = 224;

    // Output dimensions
    int outputClasses = 1000;

    // Allocate memory for input and output
    float* input;
    cudaMalloc((void**)&input, inputChannels * inputHeight * inputWidth * sizeof(float));
    float* output1;
    cudaMalloc((void**)&output1, outputClasses * sizeof(float));
    float* output2;
    cudaMalloc((void**)&output2, outputClasses * sizeof(float));

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Divide input image in half
    int halfInputHeight = inputHeight / 2;

    // Launch ResNet-18 using two streams
    resnet18(input, output1, inputChannels, halfInputHeight, inputWidth, outputClasses, stream1);
    resnet18(input + (inputChannels * halfInputHeight * inputWidth), output2, inputChannels, halfInputHeight, inputWidth, outputClasses, stream2);

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Merge the outputs of both streams into one output
    cudaMemcpy(output1 + (outputClasses / 2), output2, (outputClasses / 2) * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free allocated memory and destroy streams
    cudaFree(input);
    cudaFree(output1);
    cudaFree(output2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
