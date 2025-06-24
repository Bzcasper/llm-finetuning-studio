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
      const user = await prisma.user.findUnique({
        where: { email: session.user.email },
        include: {
          trainingJobs: {
            take: 5,
            orderBy: { createdAt: 'desc' },
            select: {
              id: true,
              modelName: true,
              status: true,
              createdAt: true,
              completedAt: true,
            }
          },
          payments: {
            take: 5,
            orderBy: { createdAt: 'desc' },
            select: {
              id: true,
              amount: true,
              currency: true,
              status: true,
              description: true,
              createdAt: true,
            }
          }
        }
      })

      if (!user) {
        return res.status(404).json({ message: 'User not found' })
      }

      res.status(200).json({
        user: {
          id: user.id,
          name: user.name,
          email: user.email,
          image: user.image,
          subscriptionStatus: user.subscriptionStatus,
          subscriptionPlan: user.subscriptionPlan,
          trainingCredits: user.trainingCredits,
          totalTrainings: user.totalTrainings,
          emailSubscribed: user.emailSubscribed,
          createdAt: user.createdAt,
        },
        recentJobs: user.trainingJobs,
        recentPayments: user.payments,
      })
    } catch (error) {
      console.error('Error fetching user profile:', error)
      res.status(500).json({ message: 'Internal server error' })
    }
  } else if (req.method === 'PUT') {
    try {
      const { name, emailSubscribed } = req.body

      const updatedUser = await prisma.user.update({
        where: { email: session.user.email },
        data: {
          ...(name && { name }),
          ...(typeof emailSubscribed === 'boolean' && { emailSubscribed }),
        }
      })

      res.status(200).json({
        user: {
          id: updatedUser.id,
          name: updatedUser.name,
          email: updatedUser.email,
          image: updatedUser.image,
          subscriptionStatus: updatedUser.subscriptionStatus,
          subscriptionPlan: updatedUser.subscriptionPlan,
          trainingCredits: updatedUser.trainingCredits,
          totalTrainings: updatedUser.totalTrainings,
          emailSubscribed: updatedUser.emailSubscribed,
        }
      })
    } catch (error) {
      console.error('Error updating user profile:', error)
      res.status(500).json({ message: 'Internal server error' })
    }
  } else {
    res.status(405).json({ message: 'Method not allowed' })
  }
}

