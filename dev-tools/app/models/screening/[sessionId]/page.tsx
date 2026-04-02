import { redirect } from "next/navigation";

export default function ScreeningSessionRedirect({ params }: { params: { sessionId: string } }) {
  redirect(`/evaluate/screening/${params.sessionId}`);
}
