import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { FileText, Cpu, Database, Zap, BarChart3, Settings, Layers, GitBranch } from "lucide-react"

const components = [
  {
    name: "gaussian_diffusion.py",
    icon: <Zap className="h-5 w-5" />,
    description: "Core diffusion process implementation",
    features: ["Forward/Reverse processes", "DDPM/DDIM sampling", "Training losses", "Beta schedules"],
    complexity: "High",
  },
  {
    name: "dist_util.py",
    icon: <Cpu className="h-5 w-5" />,
    description: "Distributed training utilities",
    features: ["MPI support", "Multi-GPU", "Parameter sync", "Device management"],
    complexity: "Medium",
  },
  {
    name: "image_datasets.py",
    icon: <Database className="h-5 w-5" />,
    description: "Image dataset loading and preprocessing",
    features: ["Recursive loading", "MPI sharding", "Smart resizing", "Class conditioning"],
    complexity: "Medium",
  },
  {
    name: "nn.py",
    icon: <Layers className="h-5 w-5" />,
    description: "Neural network utilities",
    features: ["Custom layers", "Timestep embedding", "Checkpointing", "EMA updates"],
    complexity: "Medium",
  },
  {
    name: "losses.py",
    icon: <BarChart3 className="h-5 w-5" />,
    description: "Loss function implementations",
    features: ["KL divergence", "Gaussian likelihood", "Variational bounds"],
    complexity: "Low",
  },
  {
    name: "respace.py",
    icon: <GitBranch className="h-5 w-5" />,
    description: "Timestep spacing for faster sampling",
    features: ["Custom schedules", "DDIM spacing", "Section-based"],
    complexity: "Medium",
  },
  {
    name: "resample.py",
    icon: <Settings className="h-5 w-5" />,
    description: "Importance sampling for training",
    features: ["Uniform sampling", "Loss-aware sampling", "Distributed sync"],
    complexity: "Medium",
  },
  {
    name: "logger.py",
    icon: <FileText className="h-5 w-5" />,
    description: "Comprehensive logging system",
    features: ["Multiple formats", "MPI-aware", "TensorBoard support"],
    complexity: "Low",
  },
]

const diffusionSteps = [
  {
    step: "1. Data Loading",
    description: "Load and preprocess images using image_datasets.py",
    details: "Handles recursive file discovery, smart resizing, and MPI-based data sharding",
  },
  {
    step: "2. Model Setup",
    description: "Initialize neural network with nn.py utilities",
    details: "Set up custom layers, timestep embeddings, and gradient checkpointing",
  },
  {
    step: "3. Distributed Setup",
    description: "Configure multi-GPU training with dist_util.py",
    details: "Initialize MPI, set up device mapping, and enable parameter synchronization",
  },
  {
    step: "4. Training Loop",
    description: "Train using gaussian_diffusion.py and losses.py",
    details: "Sample timesteps, add noise, predict, and compute losses with importance sampling",
  },
  {
    step: "5. Sampling",
    description: "Generate images using DDPM or DDIM sampling",
    details: "Start from noise and iteratively denoise using the trained model",
  },
]

export default function DiffusionOverview() {
  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">Diffusion Model Codebase Analysis</h1>
        <p className="text-muted-foreground">
          "Improved Denoising Diffusion Probabilistic Models" এর PyTorch Implementation
        </p>
      </div>

      <Tabs defaultValue="components" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="components">Components</TabsTrigger>
          <TabsTrigger value="process">Process Flow</TabsTrigger>
          <TabsTrigger value="math">Mathematical Foundation</TabsTrigger>
        </TabsList>

        <TabsContent value="components" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {components.map((component, index) => (
              <Card key={index} className="h-full">
                <CardHeader className="pb-3">
                  <div className="flex items-center gap-2">
                    {component.icon}
                    <CardTitle className="text-lg">{component.name}</CardTitle>
                    <Badge
                      variant={
                        component.complexity === "High"
                          ? "destructive"
                          : component.complexity === "Medium"
                            ? "default"
                            : "secondary"
                      }
                    >
                      {component.complexity}
                    </Badge>
                  </div>
                  <CardDescription>{component.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">Key Features:</h4>
                    <div className="flex flex-wrap gap-1">
                      {component.features.map((feature, idx) => (
                        <Badge key={idx} variant="outline" className="text-xs">
                          {feature}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="process" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Training & Sampling Pipeline</CardTitle>
              <CardDescription>Complete workflow from data loading to image generation</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {diffusionSteps.map((item, index) => (
                  <div key={index} className="border-l-2 border-primary pl-4 pb-4">
                    <h3 className="font-semibold text-primary">{item.step}</h3>
                    <p className="text-sm font-medium mt-1">{item.description}</p>
                    <p className="text-xs text-muted-foreground mt-1">{item.details}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="math" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Forward Process</CardTitle>
                <CardDescription>Noise addition process</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="bg-muted p-3 rounded text-sm font-mono">q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I)</div>
                <p className="text-sm text-muted-foreground">
                  Gradually adds Gaussian noise to clean images over T timesteps
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Reverse Process</CardTitle>
                <CardDescription>Denoising process</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="bg-muted p-3 rounded text-sm font-mono">
                  p(x_{"{t-1}"} | x_t) = N(μ_θ(x_t,t), Σ_θ(x_t,t))
                </div>
                <p className="text-sm text-muted-foreground">
                  Neural network learns to reverse the noise addition process
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Training Objective</CardTitle>
                <CardDescription>Loss function</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="bg-muted p-3 rounded text-sm font-mono">L = E[||ε - ε_θ(x_t, t)||²]</div>
                <p className="text-sm text-muted-foreground">Predict the noise that was added at each timestep</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Beta Schedules</CardTitle>
                <CardDescription>Noise scheduling</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="space-y-1">
                  <div className="bg-muted p-2 rounded text-xs font-mono">Linear: β_t = β_1 + (β_T - β_1) × t/T</div>
                  <div className="bg-muted p-2 rounded text-xs font-mono">
                    Cosine: ᾱ_t = cos²((t/T + s)/(1+s) × π/2)
                  </div>
                </div>
                <p className="text-sm text-muted-foreground">Controls how noise is added over time</p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      <Card>
        <CardHeader>
          <CardTitle>Key Insights</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <h4 className="font-semibold mb-2">🚀 Performance Features:</h4>
              <ul className="space-y-1 text-muted-foreground">
                <li>• MPI-based distributed training</li>
                <li>• Mixed precision (FP16) support</li>
                <li>• Gradient checkpointing for memory efficiency</li>
                <li>• Importance sampling for faster convergence</li>
                <li>• DDIM for faster sampling</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">🔧 Implementation Details:</h4>
              <ul className="space-y-1 text-muted-foreground">
                <li>• Supports both epsilon and x_start prediction</li>
                <li>• Flexible variance parameterization</li>
                <li>• Custom timestep spacing</li>
                <li>• Comprehensive logging system</li>
                <li>• Smart image preprocessing</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
