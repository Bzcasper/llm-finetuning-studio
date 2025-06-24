import { useState, useEffect } from 'react'
import { useSession, getSession } from 'next-auth/react'
import { useRouter } from 'next/router'
import { 
  CreditCard, 
  Zap, 
  Clock, 
  TrendingUp, 
  Settings, 
  Upload,
  Play,
  BarChart3,
  User,
  LogOut
} from 'lucide-react'
import Link from 'next/link'

const Dashboard = () => {
  const { data: session, status } = useSession()
  const router = useRouter()
  const [userStats, setUserStats] = useState({
    trainingCredits: 0,
    totalTrainings: 0,
    subscriptionPlan: 'free',
    subscriptionStatus: 'FREE'
  })
  const [recentJobs, setRecentJobs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/auth/signin')
    } else if (status === 'authenticated') {
      fetchUserData()
    }
  }, [status, router])

  const fetchUserData = async () => {
    try {
      const response = await fetch('/api/user/profile')
      if (response.ok) {
        const data = await response.json()
        setUserStats(data.user)
      }
    } catch (error) {
      console.error('Error fetching user data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleStartTraining = () => {
    router.push('/training/new')
  }

  if (status === 'loading' || loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  if (!session) {
    return null
  }

  const planColors = {
    free: 'bg-gray-100 text-gray-800',
    starter: 'bg-blue-100 text-blue-800',
    pro: 'bg-purple-100 text-purple-800',
    enterprise: 'bg-yellow-100 text-yellow-800'
  }

  const statusColors = {
    FREE: 'bg-gray-100 text-gray-800',
    ACTIVE: 'bg-green-100 text-green-800',
    PAST_DUE: 'bg-red-100 text-red-800',
    CANCELLED: 'bg-gray-100 text-gray-800'
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">
                LLM Fine-Tuning Studio
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">
                Welcome, {session.user?.name || 'User'}
              </span>
              <div className="flex items-center space-x-2">
                <User className="h-5 w-5 text-gray-400" />
                <button
                  onClick={() => router.push('/api/auth/signout')}
                  className="text-sm text-gray-600 hover:text-gray-900"
                >
                  <LogOut className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Zap className="h-8 w-8 text-blue-500" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Training Credits</p>
                <p className="text-2xl font-bold text-gray-900">{userStats.trainingCredits}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <TrendingUp className="h-8 w-8 text-green-500" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Trainings</p>
                <p className="text-2xl font-bold text-gray-900">{userStats.totalTrainings}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <CreditCard className="h-8 w-8 text-purple-500" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Current Plan</p>
                <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${planColors[userStats.subscriptionPlan] || planColors.free}`}>
                  {userStats.subscriptionPlan.charAt(0).toUpperCase() + userStats.subscriptionPlan.slice(1)}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Clock className="h-8 w-8 text-orange-500" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Status</p>
                <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${statusColors[userStats.subscriptionStatus] || statusColors.FREE}`}>
                  {userStats.subscriptionStatus}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <button
                  onClick={handleStartTraining}
                  className="flex items-center justify-center px-4 py-3 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                >
                  <Play className="h-5 w-5 mr-2" />
                  Start New Training
                </button>
                
                <Link href="/datasets">
                  <button className="flex items-center justify-center px-4 py-3 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                    <Upload className="h-5 w-5 mr-2" />
                    Upload Dataset
                  </button>
                </Link>
                
                <Link href="/models">
                  <button className="flex items-center justify-center px-4 py-3 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                    <BarChart3 className="h-5 w-5 mr-2" />
                    View Models
                  </button>
                </Link>
                
                <Link href="/settings">
                  <button className="flex items-center justify-center px-4 py-3 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
                    <Settings className="h-5 w-5 mr-2" />
                    Settings
                  </button>
                </Link>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Subscription</h2>
            <div className="space-y-3">
              <div>
                <p className="text-sm text-gray-600">Current Plan</p>
                <p className="font-medium">{userStats.subscriptionPlan.charAt(0).toUpperCase() + userStats.subscriptionPlan.slice(1)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Credits Remaining</p>
                <p className="font-medium">{userStats.trainingCredits}</p>
              </div>
              {userStats.subscriptionPlan === 'free' && (
                <Link href="/pricing">
                  <button className="w-full mt-4 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500">
                    Upgrade Plan
                  </button>
                </Link>
              )}
            </div>
          </div>
        </div>

        {/* Recent Training Jobs */}
        <div className="bg-white rounded-lg shadow">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Recent Training Jobs</h2>
          </div>
          <div className="p-6">
            {recentJobs.length === 0 ? (
              <div className="text-center py-8">
                <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No training jobs yet</p>
                <p className="text-sm text-gray-400 mt-1">Start your first training to see results here</p>
                <button
                  onClick={handleStartTraining}
                  className="mt-4 px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
                >
                  Start Training
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                {recentJobs.map((job, index) => (
                  <div key={index} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                    <div>
                      <p className="font-medium">{job.modelName}</p>
                      <p className="text-sm text-gray-600">{job.status}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-600">{job.createdAt}</p>
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                        job.status === 'completed' ? 'bg-green-100 text-green-800' :
                        job.status === 'running' ? 'bg-blue-100 text-blue-800' :
                        job.status === 'failed' ? 'bg-red-100 text-red-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {job.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard

export async function getServerSideProps(context) {
  const session = await getSession(context)

  if (!session) {
    return {
      redirect: {
        destination: '/auth/signin',
        permanent: false,
      },
    }
  }

  return {
    props: {
      session,
    },
  }
}

