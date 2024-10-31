package client

import (
	configv1alpha1 "github.com/NVIDIA/grove/operator/api/config/v1alpha1"
	podgangsetv1alpha1 "github.com/NVIDIA/grove/operator/api/podgangset/v1alpha1"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// Scheme is the kubernetes runtime scheme
var Scheme = runtime.NewScheme()

func init() {
	localSchemeBuilder := runtime.NewSchemeBuilder(
		configv1alpha1.AddToScheme,
		podgangsetv1alpha1.AddToScheme,
	)
	utilruntime.Must(localSchemeBuilder.AddToScheme(Scheme))
}
