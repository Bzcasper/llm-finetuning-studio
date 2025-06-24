import { useState, useEffect } from 'react'
import { useSession, signIn } from 'next-auth/react'
import { Check, X, Zap, Crown, Rocket, Star } from 'lucide-react'

const PricingPage = () => {
  const { data: session, status } = useSession()
  const [plans, setPlans] = useState([])
  const [loading, setLoading] = useState(false)
  const [currentPlan, setCurrentPlan] = useState('free')

  useEffect(() => {
    fetchPlans()
    if (session?.user) {
      setCurrentPlan(session.user.subscriptionPlan || 'free')
    }
  }, [session])

  const fetchPlans = async () => {
    try {
      const response = await fetch('/api/subscription/plans')
      const data = await response.json()
      setPlans(data.plans || [])
    } catch (error) {
      console.error('Error fetching plans:', error)
    }
  }

  const handleSubscribe = async (planId) => {
    if (!session) {
      signIn()
      return
    }

    setLoading(true)
    try {
      const response = await fetch('/api/stripe/create-checkout-session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          planId,
          successUrl: `${window.location.origin}/dashboard?success=true`,
          cancelUrl: `${window.location.origin}/pricing?cancelled=true`,
        }),
      })

      const data = await response.json()
      
      if (data.url) {
        window.location.href = data.url
      }
    } catch (error) {
      console.error('Error creating checkout session:', error)
    } finally {
      setLoading(false)
    }
  }

  const defaultPlans = [
    {
      id: 'free',
      name: 'free',
      displayName: 'Free Tier',
      description: 'Perfect for getting started with LLM fine-tuning',
      price: 0,
      interval: 'month',
      trainingCredits: 3,
      maxGpuType: 'T4',
      maxTrainingTime: 1800,
      features: [
        '3 training jobs per month',
        'T4 GPU access',
        'Basic models (up to 7B parameters)',
        'Community support',
        'Standard training time (30 min max)',
        'Basic monitoring'
      ],
      limitations: [
        'No A100/H100 access',
        'Limited model size',
        'No priority support'
      ],
      icon: Zap,
      color: 'bg-gray-100 border-gray-200',
      buttonColor: 'bg-gray-600 hover:bg-gray-700',
      popular: false
    },
    {
      id: 'starter',
      name: 'starter',
      displayName: 'Starter',
      description: 'Ideal for individual researchers and small projects',
      price: 29,
      interval: 'month',
      trainingCredits: 15,
      maxGpuType: 'A100',
      maxTrainingTime: 7200,
      features: [
        '15 training jobs per month',
        'A100 GPU access',
        'Models up to 13B parameters',
        'Email support',
        'Extended training time (2 hours max)',
        'Advanced monitoring',
        'Custom datasets',
        'LoRA & QLoRA support'
      ],
      limitations: [
        'No H100 access',
        'Standard support response time'
      ],
      icon: Rocket,
      color: 'bg-blue-50 border-blue-200',
      buttonColor: 'bg-blue-600 hover:bg-blue-700',
      popular: true
    },
    {
      id: 'pro',
      name: 'pro',
      displayName: 'Professional',
      description: 'For teams and advanced research projects',
      price: 99,
      interval: 'month',
      trainingCredits: 60,
      maxGpuType: 'H100',
      maxTrainingTime: 14400,
      features: [
        '60 training jobs per month',
        'H100 GPU access',
        'Models up to 70B parameters',
        'Priority support',
        'Extended training time (4 hours max)',
        'Advanced monitoring & analytics',
        'Custom datasets & models',
        'All fine-tuning techniques',
        'API access',
        'Team collaboration'
      ],
      limitations: [
        'Fair usage policy applies'
      ],
      icon: Crown,
      color: 'bg-purple-50 border-purple-200',
      buttonColor: 'bg-purple-600 hover:bg-purple-700',
      popular: false
    },
    {
      id: 'enterprise',
      name: 'enterprise',
      displayName: 'Enterprise',
      description: 'Custom solutions for large organizations',
      price: 'Custom',
      interval: 'month',
      trainingCredits: 'Unlimited',
      maxGpuType: 'H200/B200',
      maxTrainingTime: 'Unlimited',
      features: [
        'Unlimited training jobs',
        'Latest GPU access (H200, B200)',
        'Unlimited model sizes',
        'Dedicated support team',
        'Unlimited training time',
        'Custom monitoring solutions',
        'On-premise deployment',
        'Custom integrations',
        'SLA guarantees',
        'Advanced security features'
      ],
      limitations: [],
      icon: Star,
      color: 'bg-gradient-to-br from-yellow-50 to-orange-50 border-yellow-200',
      buttonColor: 'bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-700 hover:to-orange-700',
      popular: false
    }
  ]

  const plansToShow = plans.length > 0 ? plans : defaultPlans

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Choose Your Fine-Tuning Plan
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Scale your LLM fine-tuning with flexible plans designed for every need.
            From individual researchers to enterprise teams.
          </p>
        </div>

        {/* Current Plan Indicator */}
        {session && (
          <div className="text-center mb-8">
            <div className="inline-flex items-center px-4 py-2 bg-blue-100 text-blue-800 rounded-full">
              <span className="font-medium">
                Current Plan: {currentPlan.charAt(0).toUpperCase() + currentPlan.slice(1)}
              </span>
            </div>
          </div>
        )}

        {/* Pricing Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
          {plansToShow.map((plan) => {
            const Icon = plan.icon
            const isCurrentPlan = currentPlan === plan.name
            const isFree = plan.name === 'free'
            
            return (
              <div
                key={plan.id}
                className={`relative rounded-2xl p-8 shadow-lg transition-all duration-300 hover:shadow-xl ${plan.color} ${
                  plan.popular ? 'ring-2 ring-blue-500 scale-105' : ''
                }`}
              >
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                    <span className="bg-blue-500 text-white px-4 py-1 rounded-full text-sm font-medium">
                      Most Popular
                    </span>
                  </div>
                )}

                <div className="text-center">
                  <Icon className="h-12 w-12 mx-auto mb-4 text-gray-700" />
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">
                    {plan.displayName}
                  </h3>
                  <p className="text-gray-600 mb-6">
                    {plan.description}
                  </p>

                  <div className="mb-6">
                    {typeof plan.price === 'number' ? (
                      <>
                        <span className="text-4xl font-bold text-gray-900">
                          ${plan.price}
                        </span>
                        <span className="text-gray-600">/{plan.interval}</span>
                      </>
                    ) : (
                      <span className="text-4xl font-bold text-gray-900">
                        {plan.price}
                      </span>
                    )}
                  </div>

                  <div className="mb-6">
                    <div className="text-sm text-gray-600 mb-2">Training Credits</div>
                    <div className="text-2xl font-bold text-blue-600">
                      {plan.trainingCredits}
                      {typeof plan.trainingCredits === 'number' && '/month'}
                    </div>
                  </div>

                  <button
                    onClick={() => handleSubscribe(plan.id)}
                    disabled={loading || isCurrentPlan || (isFree && !session)}
                    className={`w-full py-3 px-6 rounded-lg font-medium transition-colors ${
                      isCurrentPlan
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        : plan.buttonColor + ' text-white'
                    }`}
                  >
                    {loading ? (
                      'Processing...'
                    ) : isCurrentPlan ? (
                      'Current Plan'
                    ) : isFree ? (
                      session ? 'Current Plan' : 'Sign Up Free'
                    ) : plan.name === 'enterprise' ? (
                      'Contact Sales'
                    ) : (
                      'Upgrade Now'
                    )}
                  </button>
                </div>

                {/* Features */}
                <div className="mt-8">
                  <h4 className="font-semibold text-gray-900 mb-4">Features:</h4>
                  <ul className="space-y-2">
                    {plan.features.map((feature, index) => (
                      <li key={index} className="flex items-start">
                        <Check className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-gray-700">{feature}</span>
                      </li>
                    ))}
                  </ul>

                  {plan.limitations.length > 0 && (
                    <div className="mt-4">
                      <h4 className="font-semibold text-gray-900 mb-2">Limitations:</h4>
                      <ul className="space-y-1">
                        {plan.limitations.map((limitation, index) => (
                          <li key={index} className="flex items-start">
                            <X className="h-4 w-4 text-red-400 mr-2 mt-0.5 flex-shrink-0" />
                            <span className="text-xs text-gray-600">{limitation}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>

        {/* FAQ Section */}
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
            Frequently Asked Questions
          </h2>
          
          <div className="space-y-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                What are training credits?
              </h3>
              <p className="text-gray-600">
                Training credits are used to start fine-tuning jobs. Each job typically uses 1 credit, 
                but larger models or longer training sessions may use more credits.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Can I change my plan anytime?
              </h3>
              <p className="text-gray-600">
                Yes, you can upgrade or downgrade your plan at any time. Changes take effect 
                immediately for upgrades, or at the next billing cycle for downgrades.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                What GPU types are available?
              </h3>
              <p className="text-gray-600">
                We offer T4, A100, H100, H200, and B200 GPUs depending on your plan. 
                Higher-tier plans get access to more powerful and newer GPU types.
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Do unused credits roll over?
              </h3>
              <p className="text-gray-600">
                Credits expire at the end of each billing cycle and do not roll over. 
                We recommend choosing a plan that matches your expected usage.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PricingPage

