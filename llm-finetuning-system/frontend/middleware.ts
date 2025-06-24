import { withAuth } from "next-auth/middleware"

export default withAuth(
  function middleware(req) {
    // Add any additional middleware logic here
    console.log("Middleware executed for:", req.nextUrl.pathname)
  },
  {
    callbacks: {
      authorized: ({ token, req }) => {
        // Check if user is authenticated for protected routes
        if (req.nextUrl.pathname.startsWith('/dashboard')) {
          return !!token
        }
        
        // Check subscription status for premium features
        if (req.nextUrl.pathname.startsWith('/training')) {
          return !!token && (
            token.subscriptionStatus === 'ACTIVE' || 
            token.subscriptionPlan === 'free'
          )
        }
        
        return true
      },
    },
  }
)

export const config = {
  matcher: [
    '/dashboard/:path*',
    '/training/:path*',
    '/api/training/:path*',
    '/api/user/:path*'
  ]
}

