import { redirect } from "next/navigation";

export default function OverlaySessionRedirect({ params }: { params: { sessionId: string } }) {
  redirect(`/evaluate/overlay/${params.sessionId}`);
}
