var sampleRate = 44100;
var blob = null;

// Downsample the buffer
function downsampleBuffer(buffer) {
    if (16000 === sampleRate) {
    return buffer;
    }
    var sampleRateRatio = sampleRate / 16000;
    var newLength = Math.round(buffer.length / sampleRateRatio);
    var result = new Float32Array(newLength);
    var offsetResult = 0;
    var offsetBuffer = 0;
    while (offsetResult < result.length) {
        var nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        var accum = 0,
        count = 0;
        for (var i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
        result[offsetResult] = accum / count;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
  }

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
  
function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  }

function flattenArray(channelBuffer, recordingLength) {
    var result = new Float32Array(recordingLength);
    var offset = 0;
    for (var i = 0; i < channelBuffer.length; i++) {
        var buffer = channelBuffer[i];
        result.set(buffer, offset);
        offset += buffer.length;
    }
    return result;
}

export function generateWaveFile(leftchannel,recordingLength) {
    var leftBuffer = flattenArray(leftchannel, recordingLength); 
    var downsampledBuffer = downsampleBuffer(leftBuffer, 16000);
    
    var buffer = new ArrayBuffer(44 + downsampledBuffer.length * 2);
    var view = new DataView(buffer);

    // RIFF chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 32 + downsampledBuffer.length * 2, true);
    writeString(view, 8, 'WAVE');
    // FMT sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // chunkSize
    view.setUint16(20, 1, true); // wFormatTag
    view.setUint16(22, 1, true); // wChannels: stereo (2 channels)
    view.setUint32(24, 16000, true); // dwSamplesPerSec
    view.setUint32(28, 16000 * 2, true); // dwAvgBytesPerSec
    view.setUint16(32, 2, true); // wBlockAlign
    view.setUint16(34, 16, true); // wBitsPerSample
    // data sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, downsampledBuffer.length * 2, true);

    floatTo16BitPCM(view, 44, downsampledBuffer);

    // write the PCM samples
    
    // our final blob
    blob = new Blob([view], { type: 'audio/wav' });
    var formData = new FormData();
    formData.append("audio", blob, "audio.wav");
    return formData;
}

export function emotionProbDict(data){
    if (data==0){
        return [{ emotion: 'Happy', probability: 0.0 }, { emotion: 'Sad', probability: 0.0 }, { emotion: 'Neutral', probability: 0.0 }, { emotion: 'Angry', probability: 0.0 }, { emotion: 'Excited', probability: 0.0 },{ emotion: 'Frustrated', probability: 0.0 }];
    } else {
        data = data["prediction"];
        const responsePredictions = [{ emotion: 'Happy', probability: data[0] },
                                    { emotion: 'Sad', probability: data[1] },
                                    { emotion: 'Neutral', probability: data[2] },
                                    { emotion: 'Angry', probability: data[3] },
                                    { emotion: 'Excited', probability: data[4] },
                                    { emotion: 'Frustrated', probability: data[5] }];
        return responsePredictions;
    }
    
}

export function distressProbDict(data){
    if (data==1000){
        return [{ status: 'Normal', probability: 0.0 }, { status: 'Distress', probability: 0.0 }];
    } else {
        const responsePredictions = [{ status: 'Normal', probability: data.prediction[0] },
                                    { status: 'Distress', probability: data.prediction[1] }];
        return responsePredictions;
    }
    
}