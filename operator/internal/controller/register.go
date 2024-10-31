package controller

import (
	configv1alpha1 "github.com/NVIDIA/grove/operator/api/config/v1alpha1"
	"github.com/NVIDIA/grove/operator/internal/controller/podgangset"

	ctrl "sigs.k8s.io/controller-runtime"
)

// RegisterControllers registers all controllers with the manager.
func RegisterControllers(mgr ctrl.Manager, controllerConfig configv1alpha1.ControllerConfiguration) error {
	pgsReconciler := podgangset.NewReconciler(mgr, controllerConfig.PodGangSet)
	if err := pgsReconciler.RegisterWithManager(mgr); err != nil {
		return err
	}
	return nil
}
