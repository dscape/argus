import { redirect } from "next/navigation";

export default function PhysicalTrainAnnotationIndexPage() {
  redirect("/annotate/physical?split=train");
}
