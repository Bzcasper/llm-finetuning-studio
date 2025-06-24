import NextAuth from 'next-auth'
import GoogleProvider from 'next-auth/providers/google'
import GitHubProvider from 'next-auth/providers/github'
import CredentialsProvider from 'next-auth/providers/credentials'
import { PrismaAdapter } from "@next-auth/prisma-adapter"
import { prisma } from '../../../lib/prisma'
import bcrypt from 'bcryptjs'

export default NextAuth({
  adapter: PrismaAdapter(prisma),
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    }),
    GitHubProvider({
      clientId: process.env.GITHUB_ID,
      clientSecret: process.env.GITHUB_SECRET,
    }),
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null
        }

        const user = await prisma.user.findUnique({
          where: {
            email: credentials.email
          }
        })

        if (!user) {
          return null
        }

        const isPasswordValid = await bcrypt.compare(
          credentials.password,
          user.password
        )

        if (!isPasswordValid) {
          return null
        }

        return {
          id: user.id,
          email: user.email,
          name: user.name,
          subscriptionStatus: user.subscriptionStatus,
          subscriptionPlan: user.subscriptionPlan,
        }
      }
    })
  ],
  session: {
    strategy: "jwt",
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.subscriptionStatus = user.subscriptionStatus
        token.subscriptionPlan = user.subscriptionPlan
      }
      return token
    },
    async session({ session, token }) {
      if (token) {
        session.user.id = token.sub
        session.user.subscriptionStatus = token.subscriptionStatus
        session.user.subscriptionPlan = token.subscriptionPlan
      }
      return session
    },
    async signIn({ user, account, profile }) {
      if (account?.provider === "google" || account?.provider === "github") {
        try {
          const existingUser = await prisma.user.findUnique({
            where: { email: user.email }
          })

          if (!existingUser) {
            const newUser = await prisma.user.create({
              data: {
                email: user.email,
                name: user.name,
                image: user.image,
                subscriptionStatus: 'FREE',
                subscriptionPlan: 'free',
                trainingCredits: 3, // Free tier gets 3 training credits
                emailSubscribed: true,
              }
            })

            // Send welcome email
            try {
              const EmailService = require('../../../lib/email').default
              const emailService = EmailService.getInstance()
              await emailService.sendWelcomeEmail(user.email, user.name || 'User')
              console.log(`Welcome email sent to ${user.email}`)
            } catch (error) {
              console.error('Failed to send welcome email:', error)
            }
          }
        } catch (error) {
          console.error("Error creating user:", error)
          return false
        }
      }
      return true
    }
  },
  pages: {
    signIn: '/auth/signin',
    signUp: '/auth/signup',
    error: '/auth/error',
  },
  secret: process.env.NEXTAUTH_SECRET,
})

