import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '../auth/[...nextauth]'
import { prisma } from '../../../lib/prisma'

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const session = await getServerSession(req, res, authOptions)
  
  if (!session?.user?.email) {
    return res.status(401).json({ message: 'Unauthorized' })
  }

  if (req.method === 'GET') {
    try {
      // Get subscription plans
      const plans = await prisma.subscriptionPlan.findMany({
        where: { active: true },
        orderBy: { price: 'asc' }
      })

      // Get user's current subscription info
      const user = await prisma.user.findUnique({
        where: { email: session.user.email },
        select: {
          subscriptionStatus: true,
          subscriptionPlan: true,
          trainingCredits: true,
          totalTrainings: true,
          createdAt: true
        }
      })

      res.status(200).json({
        plans,
        currentSubscription: user
      })
    } catch (error) {
      console.error('Error fetching subscription data:', error)
      res.status(500).json({ message: 'Internal server error' })
    }
  } else if (req.method === 'POST') {
    // Handle subscription updates
    try {
      const { action, planId } = req.body

      const user = await prisma.user.findUnique({
        where: { email: session.user.email }
      })

      if (!user) {
        return res.status(404).json({ message: 'User not found' })
      }

      switch (action) {
        case 'cancel':
          // Cancel subscription logic would go here
          // This would typically involve calling Stripe API
          res.status(200).json({ message: 'Subscription cancelled' })
          break
        
        case 'reactivate':
          // Reactivate subscription logic
          res.status(200).json({ message: 'Subscription reactivated' })
          break
        
        default:
          res.status(400).json({ message: 'Invalid action' })
      }
    } catch (error) {
      console.error('Error updating subscription:', error)
      res.status(500).json({ message: 'Internal server error' })
    }
  } else {
    res.status(405).json({ message: 'Method not allowed' })
  }
}

