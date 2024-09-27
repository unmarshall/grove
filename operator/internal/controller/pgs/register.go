package pgs

import (
	"github.com/NVIDIA/grove/operator/api/v1alpha1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

const controllerName = "podgangset-controller"

func (r *Reconciler) RegisterWithManager(mgr manager.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		Named(controllerName).
		WithOptions(controller.Options{}).
		For(&v1alpha1.PodGangSet{}).
		Complete(r)
}
