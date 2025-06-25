import { useSession } from "next-auth/react";

export default function Component() {
  const { data } = useSession();
  const accessToken = data?.accessToken;

  return <div>Access Token: {accessToken ?? "Not available"}</div>;
}
