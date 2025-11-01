import { motion } from "framer-motion";
import { Upload, Sparkles, BarChart3, BookOpen } from "lucide-react";
import { Navigation } from "@/components/Navigation";
import { Footer } from "@/components/Footer";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

const steps = [
  {
    icon: Upload,
    title: "Upload or Describe Requirements",
    description: "Drag & drop your requirements document or simply describe your application needs in plain text. We support .txt, .pdf, and .docx files.",
    color: "from-blue-500 to-cyan-500",
  },
  {
    icon: Sparkles,
    title: "AI Analysis",
    description: "Our advanced AI powered by Gemini analyzes your requirements, understanding context, dependencies, and best practices for cloud architecture.",
    color: "from-purple-500 to-pink-500",
  },
  {
    icon: BarChart3,
    title: "Instant AWS Architecture & Cost",
    description: "Receive a comprehensive architecture diagram with recommended AWS services, confidence scores, and estimated monthly costs within seconds.",
    color: "from-orange-500 to-red-500",
  },
  {
    icon: BookOpen,
    title: "View Deployment Guide & Save",
    description: "Access step-by-step deployment instructions and save your architecture to your personal library for future reference and regeneration.",
    color: "from-teal-500 to-green-500",
  },
];

const HowItWorks = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col">
      <Navigation />
      
      <main className="flex-1 py-20 px-4">
        <div className="container mx-auto max-w-6xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h1 className="text-5xl md:text-6xl font-bold mb-6 gradient-text">
              How It Works
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Transform your application requirements into production-ready cloud architecture in just four simple steps
            </p>
          </motion.div>

          {/* Desktop: Horizontal Stepper */}
          <div className="hidden md:block">
            <div className="relative">
              {/* Connecting Line */}
              <div className="absolute top-20 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 via-purple-500 via-orange-500 to-teal-500 opacity-20" />
              
              <div className="grid grid-cols-4 gap-8">
                {steps.map((step, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: index * 0.2 }}
                    className="relative"
                  >
                    <div className="flex flex-col items-center text-center">
                      <div className={`w-40 h-40 rounded-full bg-gradient-to-br ${step.color} p-1 mb-6 animate-glow`}>
                        <div className="w-full h-full rounded-full bg-background flex items-center justify-center">
                          <step.icon className="w-16 h-16 text-primary" />
                        </div>
                      </div>
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary to-accent text-white flex items-center justify-center font-bold mb-4">
                        {index + 1}
                      </div>
                      <h3 className="text-xl font-bold mb-3">{step.title}</h3>
                      <p className="text-muted-foreground text-sm leading-relaxed">
                        {step.description}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>

          {/* Mobile: Vertical Stepper */}
          <div className="md:hidden space-y-8">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                className="relative pl-12"
              >
                <div className="absolute left-0 top-0 w-8 h-8 rounded-full bg-gradient-to-br from-primary to-accent text-white flex items-center justify-center font-bold">
                  {index + 1}
                </div>
                {index < steps.length - 1 && (
                  <div className="absolute left-4 top-8 bottom-0 w-0.5 bg-gradient-to-b from-primary to-accent opacity-20" />
                )}
                <div className="glass-card rounded-2xl p-6">
                  <div className={`w-16 h-16 rounded-full bg-gradient-to-br ${step.color} p-1 mb-4`}>
                    <div className="w-full h-full rounded-full bg-background flex items-center justify-center">
                      <step.icon className="w-8 h-8 text-primary" />
                    </div>
                  </div>
                  <h3 className="text-xl font-bold mb-3">{step.title}</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    {step.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.8 }}
            className="mt-16 text-center"
          >
            <Button
              size="lg"
              onClick={() => navigate("/")}
              className="text-lg px-8 py-6 hover:scale-105 transition-transform"
            >
              Try It Now
            </Button>
          </motion.div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default HowItWorks;
