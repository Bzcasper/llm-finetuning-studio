import { useState, useEffect, useCallback } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Input } from '@/components/ui/input.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Switch } from '@/components/ui/switch.jsx'
import { Separator } from '@/components/ui/separator.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import { 
  Upload, 
  Play, 
  Settings, 
  Database, 
  Brain, 
  BarChart3, 
  Download,
  FileText,
  Cpu,
  Zap,
  Eye,
  Key,
  Server,
  CheckCircle,
  XCircle,
  Clock,
  Loader2
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import './App.css'

// API configuration
const API_BASE_URL = 'http://localhost:8000/api'

function App() {
  const [activeTab, setActiveTab] = useState('configuration')
  const [currentJobId, setCurrentJobId] = useState(null)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [trainingLogs, setTrainingLogs] = useState([])
  const [availableDatasets, setAvailableDatasets] = useState([])
  const [availableModels, setAvailableModels] = useState([])
  const [isConnected, setIsConnected] = useState(false)
  
  // Training configuration state
  const [config, setConfig] = useState({
    model_name: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    dataset_path: '',
    output_dir: '/vol/finetuned_model',
    lora_r: 8,
    lora_alpha: 16,
    lora_dropout: 0.05,
    learning_rate: 0.0002,
    num_train_epochs: 3,
    per_device_train_batch_size: 2,
    gradient_accumulation_steps: 1,
    optimizer_type: 'adamw_torch',
    use_4bit_quantization: false,
    gpu_type: 'A100',
    timeout: 3600
  })

  // Modal credentials state
  const [modalConfig, setModalConfig] = useState({
    secretId: '',
    apiKey: '',
    token: '',
    hfToken: '',
    wandbKey: ''
  })

  // Training metrics for visualization
  const [trainingMetrics, setTrainingMetrics] = useState([])

  // API functions
  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`)
      if (response.ok) {
        setIsConnected(true)
      } else {
        setIsConnected(false)
      }
    } catch (error) {
      setIsConnected(false)
    }
  }, [])

  const fetchDatasets = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/datasets`)
      if (response.ok) {
        const data = await response.json()
        setAvailableDatasets(data.datasets)
      }
    } catch (error) {
      console.error('Failed to fetch datasets:', error)
    }
  }, [])

  const fetchModels = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/models`)
      if (response.ok) {
        const data = await response.json()
        setAvailableModels(data.models)
      }
    } catch (error) {
      console.error('Failed to fetch models:', error)
    }
  }, [])

  const startTraining = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/training/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      })
      
      if (response.ok) {
        const data = await response.json()
        setCurrentJobId(data.job_id)
        setActiveTab('training')
      } else {
        console.error('Failed to start training')
      }
    } catch (error) {
      console.error('Error starting training:', error)
    }
  }

  const fetchTrainingStatus = useCallback(async (jobId) => {
    if (!jobId) return
    
    try {
      const response = await fetch(`${API_BASE_URL}/training/status/${jobId}`)
      if (response.ok) {
        const status = await response.json()
        setTrainingStatus(status)
        
        // Update metrics for visualization
        if (status.loss !== null && status.accuracy !== null) {
          setTrainingMetrics(prev => {
            const newMetrics = [...prev]
            const existingIndex = newMetrics.findIndex(m => m.epoch === status.current_epoch)
            
            if (existingIndex >= 0) {
              newMetrics[existingIndex] = {
                epoch: status.current_epoch,
                loss: status.loss,
                accuracy: status.accuracy,
                gpu_utilization: status.gpu_utilization,
                gpu_memory: status.gpu_memory_used
              }
            } else {
              newMetrics.push({
                epoch: status.current_epoch,
                loss: status.loss,
                accuracy: status.accuracy,
                gpu_utilization: status.gpu_utilization,
                gpu_memory: status.gpu_memory_used
              })
            }
            
            return newMetrics.sort((a, b) => a.epoch - b.epoch)
          })
        }
      }
    } catch (error) {
      console.error('Error fetching training status:', error)
    }
  }, [])

  const fetchTrainingLogs = useCallback(async (jobId) => {
    if (!jobId) return
    
    try {
      const response = await fetch(`${API_BASE_URL}/training/logs/${jobId}`)
      if (response.ok) {
        const data = await response.json()
        setTrainingLogs(data.logs)
      }
    } catch (error) {
      console.error('Error fetching training logs:', error)
    }
  }, [])

  // Effect hooks
  useEffect(() => {
    checkHealth()
    fetchDatasets()
    fetchModels()
    
    // Set up periodic health checks
    const healthInterval = setInterval(checkHealth, 30000)
    return () => clearInterval(healthInterval)
  }, [checkHealth, fetchDatasets, fetchModels])

  useEffect(() => {
    if (currentJobId) {
      // Poll for training status and logs
      const statusInterval = setInterval(() => {
        fetchTrainingStatus(currentJobId)
        fetchTrainingLogs(currentJobId)
      }, 2000)
      
      return () => clearInterval(statusInterval)
    }
  }, [currentJobId, fetchTrainingStatus, fetchTrainingLogs])

  const handleConfigChange = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }

  const handleModalConfigChange = (key, value) => {
    setModalConfig(prev => ({ ...prev, [key]: value }))
  }

  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      setConfig(prev => ({ ...prev, dataset_path: file.name }))
    }
  }

  const formatTime = (seconds) => {
    if (!seconds) return 'N/A'
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />
      case 'running':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
      default:
        return <Clock className="w-4 h-4 text-yellow-500" />
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto p-6">
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
                LLM Fine-Tuning Studio
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-300">
                Professional fine-tuning platform powered by Modal.com
              </p>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-600">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="configuration" className="flex items-center gap-2">
              <Settings className="w-4 h-4" />
              Configuration
            </TabsTrigger>
            <TabsTrigger value="dataset" className="flex items-center gap-2">
              <Database className="w-4 h-4" />
              Dataset
            </TabsTrigger>
            <TabsTrigger value="training" className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Training
              {trainingStatus && (
                <Badge variant={trainingStatus.status === 'running' ? 'default' : 'secondary'}>
                  {trainingStatus.status}
                </Badge>
              )}
            </TabsTrigger>
            <TabsTrigger value="monitoring" className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Monitoring
            </TabsTrigger>
            <TabsTrigger value="credentials" className="flex items-center gap-2">
              <Key className="w-4 h-4" />
              Credentials
            </TabsTrigger>
          </TabsList>

          <TabsContent value="configuration" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    Model Configuration
                  </CardTitle>
                  <CardDescription>
                    Configure the base model and output settings
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="modelName">Model Name/Path</Label>
                    <Select value={config.model_name} onValueChange={(value) => handleConfigChange('model_name', value)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {availableModels.map((model) => (
                          <SelectItem key={model.name} value={model.name}>
                            {model.name} ({model.type})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="outputDir">Output Directory</Label>
                    <Input
                      id="outputDir"
                      value={config.output_dir}
                      onChange={(e) => handleConfigChange('output_dir', e.target.value)}
                      placeholder="/vol/finetuned_model"
                    />
                  </div>
                  <div>
                    <Label htmlFor="gpuType">GPU Type</Label>
                    <Select value={config.gpu_type} onValueChange={(value) => handleConfigChange('gpu_type', value)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="T4">T4</SelectItem>
                        <SelectItem value="L4">L4</SelectItem>
                        <SelectItem value="A10G">A10G</SelectItem>
                        <SelectItem value="A100">A100-40GB</SelectItem>
                        <SelectItem value="A100-80GB">A100-80GB</SelectItem>
                        <SelectItem value="L40S">L40S</SelectItem>
                        <SelectItem value="H100">H100</SelectItem>
                        <SelectItem value="H200">H200</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="w-5 h-5" />
                    LoRA Configuration
                  </CardTitle>
                  <CardDescription>
                    Low-Rank Adaptation parameters for efficient fine-tuning
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="loraR">LoRA Rank (r)</Label>
                      <Input
                        id="loraR"
                        type="number"
                        value={config.lora_r}
                        onChange={(e) => handleConfigChange('lora_r', parseInt(e.target.value))}
                      />
                    </div>
                    <div>
                      <Label htmlFor="loraAlpha">LoRA Alpha</Label>
                      <Input
                        id="loraAlpha"
                        type="number"
                        value={config.lora_alpha}
                        onChange={(e) => handleConfigChange('lora_alpha', parseInt(e.target.value))}
                      />
                    </div>
                  </div>
                  <div>
                    <Label htmlFor="loraDropout">LoRA Dropout</Label>
                    <Input
                      id="loraDropout"
                      type="number"
                      step="0.01"
                      value={config.lora_dropout}
                      onChange={(e) => handleConfigChange('lora_dropout', parseFloat(e.target.value))}
                    />
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="quantization"
                      checked={config.use_4bit_quantization}
                      onCheckedChange={(checked) => handleConfigChange('use_4bit_quantization', checked)}
                    />
                    <Label htmlFor="quantization">Enable 4-bit Quantization (QLoRA)</Label>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Cpu className="w-5 h-5" />
                    Training Parameters
                  </CardTitle>
                  <CardDescription>
                    Configure learning rate, batch size, and optimization
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="learningRate">Learning Rate</Label>
                      <Input
                        id="learningRate"
                        type="number"
                        step="0.0001"
                        value={config.learning_rate}
                        onChange={(e) => handleConfigChange('learning_rate', parseFloat(e.target.value))}
                      />
                    </div>
                    <div>
                      <Label htmlFor="epochs">Training Epochs</Label>
                      <Input
                        id="epochs"
                        type="number"
                        value={config.num_train_epochs}
                        onChange={(e) => handleConfigChange('num_train_epochs', parseInt(e.target.value))}
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="batchSize">Batch Size</Label>
                      <Input
                        id="batchSize"
                        type="number"
                        value={config.per_device_train_batch_size}
                        onChange={(e) => handleConfigChange('per_device_train_batch_size', parseInt(e.target.value))}
                      />
                    </div>
                    <div>
                      <Label htmlFor="gradientSteps">Gradient Accumulation</Label>
                      <Input
                        id="gradientSteps"
                        type="number"
                        value={config.gradient_accumulation_steps}
                        onChange={(e) => handleConfigChange('gradient_accumulation_steps', parseInt(e.target.value))}
                      />
                    </div>
                  </div>
                  <div>
                    <Label htmlFor="optimizer">Optimizer</Label>
                    <Select value={config.optimizer_type} onValueChange={(value) => handleConfigChange('optimizer_type', value)}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="adamw_torch">AdamW (PyTorch)</SelectItem>
                        <SelectItem value="adamw_hf">AdamW (HuggingFace)</SelectItem>
                        <SelectItem value="sgd">SGD</SelectItem>
                        <SelectItem value="adafactor">Adafactor</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Server className="w-5 h-5" />
                    System Configuration
                  </CardTitle>
                  <CardDescription>
                    Timeout and resource allocation settings
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="timeout">Timeout (seconds)</Label>
                    <Input
                      id="timeout"
                      type="number"
                      value={config.timeout}
                      onChange={(e) => handleConfigChange('timeout', parseInt(e.target.value))}
                    />
                  </div>
                  <Alert>
                    <AlertDescription>
                      Current configuration will use {config.gpu_type} GPU with {config.use_4bit_quantization ? '4-bit' : 'full'} precision training.
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="dataset" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="w-5 h-5" />
                  Dataset Management
                </CardTitle>
                <CardDescription>
                  Upload, select, or configure your training dataset
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card className="p-4">
                    <div className="text-center">
                      <Upload className="w-8 h-8 mx-auto mb-2 text-blue-500" />
                      <h3 className="font-semibold mb-2">Upload Dataset</h3>
                      <Input
                        type="file"
                        accept=".json,.jsonl,.csv,.txt"
                        onChange={handleFileUpload}
                        className="mb-2"
                      />
                      <p className="text-sm text-gray-500">
                        Supports JSON, JSONL, CSV, TXT
                      </p>
                    </div>
                  </Card>
                  
                  <Card className="p-4">
                    <div className="text-center">
                      <FileText className="w-8 h-8 mx-auto mb-2 text-green-500" />
                      <h3 className="font-semibold mb-2">HuggingFace Dataset</h3>
                      <Input
                        placeholder="dataset/name"
                        value={config.dataset_path}
                        onChange={(e) => handleConfigChange('dataset_path', e.target.value)}
                        className="mb-2"
                      />
                      <p className="text-sm text-gray-500">
                        Use HuggingFace dataset ID
                      </p>
                    </div>
                  </Card>
                  
                  <Card className="p-4">
                    <div className="text-center">
                      <Download className="w-8 h-8 mx-auto mb-2 text-purple-500" />
                      <h3 className="font-semibold mb-2">Volume Storage</h3>
                      <Select onValueChange={(value) => handleConfigChange('dataset_path', value)}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select from volume" />
                        </SelectTrigger>
                        <SelectContent>
                          {availableDatasets.map((dataset) => (
                            <SelectItem key={dataset.path} value={dataset.path}>
                              {dataset.name} ({dataset.size})
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="text-sm text-gray-500">
                        Select from Modal volume
                      </p>
                    </div>
                  </Card>
                </div>

                <Separator />

                <div>
                  <h3 className="text-lg font-semibold mb-4">Dataset Preview</h3>
                  <Card className="p-4">
                    <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                      <p className="text-sm font-mono">
                        {config.dataset_path ? `Selected: ${config.dataset_path}` : 'No dataset selected'}
                      </p>
                      {config.dataset_path && (
                        <div className="mt-2 space-y-1">
                          <Badge variant="outline">Format: Auto-detected</Badge>
                          <Badge variant="outline">Size: Calculating...</Badge>
                          <Badge variant="outline">Columns: text, instruction, response</Badge>
                        </div>
                      )}
                    </div>
                  </Card>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="training" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Play className="w-5 h-5" />
                    Training Control
                    {trainingStatus && getStatusIcon(trainingStatus.status)}
                  </CardTitle>
                  <CardDescription>
                    Start, monitor, and control your training job
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Training Progress</span>
                      <span>{trainingStatus?.progress?.toFixed(1) || 0}%</span>
                    </div>
                    <Progress value={trainingStatus?.progress || 0} className="w-full" />
                    {trainingStatus && (
                      <div className="text-sm text-gray-600">
                        Epoch {trainingStatus.current_epoch} of {trainingStatus.total_epochs}
                        {trainingStatus.estimated_time_remaining && (
                          <span> • {formatTime(trainingStatus.estimated_time_remaining)} remaining</span>
                        )}
                      </div>
                    )}
                  </div>
                  
                  <div className="flex gap-2">
                    <Button 
                      onClick={startTraining} 
                      disabled={!isConnected || !config.dataset_path || (trainingStatus?.status === 'running')}
                      className="flex-1"
                    >
                      {trainingStatus?.status === 'running' ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Training...
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4 mr-2" />
                          Start Training
                        </>
                      )}
                    </Button>
                    <Button variant="outline" disabled={trainingStatus?.status !== 'running'}>
                      Pause
                    </Button>
                    <Button variant="destructive" disabled={trainingStatus?.status !== 'running'}>
                      Stop
                    </Button>
                  </div>

                  <Alert>
                    <AlertDescription>
                      {!isConnected 
                        ? 'Backend API is not connected'
                        : trainingStatus?.status === 'running'
                          ? `Training in progress on ${config.gpu_type} GPU...`
                          : config.dataset_path 
                            ? 'Ready to start training'
                            : 'Please select a dataset first'
                      }
                    </AlertDescription>
                  </Alert>

                  {trainingStatus && (
                    <div className="grid grid-cols-2 gap-4 pt-4">
                      <div className="text-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                        <div className="text-lg font-bold text-blue-600">
                          {trainingStatus.loss?.toFixed(4) || 'N/A'}
                        </div>
                        <div className="text-sm text-gray-600">Current Loss</div>
                      </div>
                      <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                        <div className="text-lg font-bold text-green-600">
                          {trainingStatus.accuracy?.toFixed(3) || 'N/A'}
                        </div>
                        <div className="text-sm text-gray-600">Accuracy</div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Eye className="w-5 h-5" />
                    Training Logs
                  </CardTitle>
                  <CardDescription>
                    Real-time training output and status
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="bg-black text-green-400 p-4 rounded-lg h-64 overflow-y-auto font-mono text-sm">
                    {trainingLogs.length > 0 ? (
                      trainingLogs.map((log, index) => (
                        <div key={index} className="mb-1">
                          {log}
                        </div>
                      ))
                    ) : (
                      <div className="text-gray-500">No logs yet. Start training to see output.</div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Training Summary</CardTitle>
                <CardDescription>
                  Current configuration and estimated resources
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">{config.num_train_epochs}</div>
                    <div className="text-sm text-gray-600">Epochs</div>
                  </div>
                  <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">{config.gpu_type}</div>
                    <div className="text-sm text-gray-600">GPU Type</div>
                  </div>
                  <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                    <div className="text-2xl font-bold text-purple-600">{config.lora_r}</div>
                    <div className="text-sm text-gray-600">LoRA Rank</div>
                  </div>
                  <div className="text-center p-4 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                    <div className="text-2xl font-bold text-orange-600">{Math.round(config.timeout / 60)}m</div>
                    <div className="text-sm text-gray-600">Timeout</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="monitoring" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Training Loss</CardTitle>
                  <CardDescription>
                    Loss progression over training epochs
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={trainingMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="loss" 
                        stroke="#8884d8" 
                        strokeWidth={2}
                        dot={{ fill: '#8884d8' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Training Accuracy</CardTitle>
                  <CardDescription>
                    Accuracy improvement over training epochs
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={trainingMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis />
                      <Tooltip />
                      <Line 
                        type="monotone" 
                        dataKey="accuracy" 
                        stroke="#82ca9d" 
                        strokeWidth={2}
                        dot={{ fill: '#82ca9d' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>GPU Utilization</CardTitle>
                  <CardDescription>
                    Real-time GPU usage monitoring
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={trainingMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis />
                      <Tooltip />
                      <Area 
                        type="monotone" 
                        dataKey="gpu_utilization" 
                        stroke="#ffc658" 
                        fill="#ffc658"
                        fillOpacity={0.3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>GPU Memory Usage</CardTitle>
                  <CardDescription>
                    Memory consumption over time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={trainingMetrics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis />
                      <Tooltip />
                      <Area 
                        type="monotone" 
                        dataKey="gpu_memory" 
                        stroke="#ff7300" 
                        fill="#ff7300"
                        fillOpacity={0.3}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>System Metrics</CardTitle>
                <CardDescription>
                  Real-time system performance and resource usage
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 border rounded-lg">
                    <div className="text-3xl font-bold text-blue-600 mb-2">
                      {trainingStatus?.gpu_utilization?.toFixed(1) || '0'}%
                    </div>
                    <div className="text-sm text-gray-600">GPU Utilization</div>
                    <Progress value={trainingStatus?.gpu_utilization || 0} className="mt-2" />
                  </div>
                  <div className="text-center p-4 border rounded-lg">
                    <div className="text-3xl font-bold text-green-600 mb-2">
                      {trainingStatus?.gpu_memory_used?.toFixed(1) || '0'}GB
                    </div>
                    <div className="text-sm text-gray-600">GPU Memory</div>
                    <Progress value={(trainingStatus?.gpu_memory_used || 0) * 5} className="mt-2" />
                  </div>
                  <div className="text-center p-4 border rounded-lg">
                    <div className="text-3xl font-bold text-purple-600 mb-2">
                      {trainingStatus?.estimated_time_remaining ? formatTime(trainingStatus.estimated_time_remaining) : 'N/A'}
                    </div>
                    <div className="text-sm text-gray-600">Est. Remaining</div>
                    <Progress value={trainingStatus?.progress || 0} className="mt-2" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Model Outputs</CardTitle>
                <CardDescription>
                  Download trained models and checkpoints
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 border rounded-lg">
                    <div>
                      <div className="font-medium">Final Model</div>
                      <div className="text-sm text-gray-500">finetuned_model_final.safetensors</div>
                    </div>
                    <Button size="sm" disabled={trainingStatus?.status !== 'completed'}>
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </Button>
                  </div>
                  <div className="flex items-center justify-between p-3 border rounded-lg">
                    <div>
                      <div className="font-medium">Training Logs</div>
                      <div className="text-sm text-gray-500">training_logs.txt</div>
                    </div>
                    <Button size="sm" variant="outline" disabled={!currentJobId}>
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="credentials" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Key className="w-5 h-5" />
                  Modal.com Credentials
                </CardTitle>
                <CardDescription>
                  Configure your Modal.com authentication and API access
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="secretId">Modal Secret ID</Label>
                  <Input
                    id="secretId"
                    type="password"
                    value={modalConfig.secretId}
                    onChange={(e) => handleModalConfigChange('secretId', e.target.value)}
                    placeholder="Enter your Modal secret ID"
                  />
                </div>
                <div>
                  <Label htmlFor="apiKey">Modal API Key</Label>
                  <Input
                    id="apiKey"
                    type="password"
                    value={modalConfig.apiKey}
                    onChange={(e) => handleModalConfigChange('apiKey', e.target.value)}
                    placeholder="Enter your Modal API key"
                  />
                </div>
                <div>
                  <Label htmlFor="token">Modal Token</Label>
                  <Input
                    id="token"
                    type="password"
                    value={modalConfig.token}
                    onChange={(e) => handleModalConfigChange('token', e.target.value)}
                    placeholder="Enter your Modal token"
                  />
                </div>
                <Separator />
                <div>
                  <Label htmlFor="hfToken">HuggingFace Token (Optional)</Label>
                  <Input
                    id="hfToken"
                    type="password"
                    value={modalConfig.hfToken}
                    onChange={(e) => handleModalConfigChange('hfToken', e.target.value)}
                    placeholder="Enter HuggingFace token for private models"
                  />
                </div>
                <div>
                  <Label htmlFor="wandbKey">Weights & Biases API Key (Optional)</Label>
                  <Input
                    id="wandbKey"
                    type="password"
                    value={modalConfig.wandbKey}
                    onChange={(e) => handleModalConfigChange('wandbKey', e.target.value)}
                    placeholder="Enter W&B API key for experiment tracking"
                  />
                </div>
                <Alert>
                  <AlertDescription>
                    Your credentials are stored securely and used only for authentication with Modal.com services.
                  </AlertDescription>
                </Alert>
                <Button className="w-full">
                  Save Credentials
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Connection Status</CardTitle>
                <CardDescription>
                  Verify your connection to Modal.com and other services
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                      <span>Backend API</span>
                    </div>
                    <Badge variant="outline" className={isConnected ? 'text-green-600' : 'text-red-600'}>
                      {isConnected ? 'Connected' : 'Disconnected'}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <span>Modal.com API</span>
                    </div>
                    <Badge variant="outline" className="text-yellow-600">Pending</Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
                      <span>Weights & Biases</span>
                    </div>
                    <Badge variant="outline" className="text-gray-600">Not Connected</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

export default App

